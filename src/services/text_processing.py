import re
import string
from typing import List, Dict, Any, Optional
from loguru import logger
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import asyncio
from concurrent.futures import ThreadPoolExecutor


class TextProcessor:
    """
    Advanced text processing service for review analysis
    """
    
    def __init__(self):
        self._ensure_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Common product aspects for e-commerce
        self.product_aspects = {
            'quality': ['quality', 'build', 'construction', 'material', 'durable', 'durability', 'sturdy', 'solid', 'cheap', 'flimsy'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'worth', 'affordable', 'overpriced', 'budget'],
            'shipping': ['shipping', 'delivery', 'package', 'packaging', 'arrived', 'fast', 'slow', 'delay'],
            'design': ['design', 'look', 'appearance', 'style', 'color', 'beautiful', 'ugly', 'attractive'],
            'usability': ['easy', 'difficult', 'user', 'interface', 'simple', 'complex', 'intuitive', 'confusing'],
            'size': ['size', 'big', 'small', 'large', 'tiny', 'huge', 'compact', 'dimensions', 'fit'],
            'performance': ['performance', 'speed', 'fast', 'slow', 'efficient', 'battery', 'power'],
            'customer_service': ['service', 'support', 'help', 'staff', 'representative', 'response']
        }
        
        # Sentiment keywords
        self.positive_words = {
            'excellent', 'amazing', 'great', 'fantastic', 'wonderful', 'perfect', 'awesome', 'outstanding',
            'superb', 'brilliant', 'marvelous', 'good', 'nice', 'pleased', 'satisfied', 'happy',
            'love', 'like', 'recommend', 'impressed', 'beautiful', 'fast', 'easy', 'comfortable'
        }
        
        self.negative_words = {
            'terrible', 'awful', 'horrible', 'bad', 'poor', 'worst', 'hate', 'disappointed',
            'frustrating', 'annoying', 'useless', 'broken', 'defective', 'cheap', 'flimsy',
            'slow', 'difficult', 'uncomfortable', 'ugly', 'expensive', 'waste', 'regret'
        }
    
    def _ensure_nltk_data(self):
        """
        Download required NLTK data if not present
        """
        required_data = [
            'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
            'maxent_ne_chunker', 'words', 'omw-1.4'
        ]
        
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                try:
                    nltk.download(data, quiet=True)
                except Exception as e:
                    logger.warning(f"Could not download NLTK data {data}: {e}")
    
    async def preprocess_text(self, text: str, for_sentiment: bool = True) -> Dict[str, Any]:
        """
        Comprehensive text preprocessing
        """
        loop = asyncio.get_event_loop()
        
        def _preprocess():
            result = {
                'original': text,
                'cleaned': '',
                'tokens': [],
                'lemmatized': [],
                'sentences': [],
                'word_count': 0,
                'sentence_count': 0,
                'readability_score': 0.0
            }
            
            # Basic cleaning
            cleaned = self._basic_clean(text)
            result['cleaned'] = cleaned
            
            if not cleaned:
                return result
            
            # Tokenization
            tokens = word_tokenize(cleaned.lower())
            result['tokens'] = tokens
            result['word_count'] = len(tokens)
            
            # Remove stopwords for sentiment analysis
            if for_sentiment:
                tokens = [word for word in tokens if word not in self.stop_words]
            
            # Lemmatization
            lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
            result['lemmatized'] = lemmatized
            
            # Sentence tokenization
            sentences = sent_tokenize(cleaned)
            result['sentences'] = sentences
            result['sentence_count'] = len(sentences)
            
            # Calculate readability score (simplified)
            result['readability_score'] = self._calculate_readability(text, sentences, tokens)
            
            return result
        
        return await loop.run_in_executor(self.executor, _preprocess)
    
    def _basic_clean(self, text: str) -> str:
        """
        Basic text cleaning
        """
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _calculate_readability(self, text: str, sentences: List[str], words: List[str]) -> float:
        """
        Calculate a simplified readability score (0-100, higher is more readable)
        """
        if not text or not sentences or not words:
            return 0.0
        
        try:
            avg_sentence_length = len(words) / len(sentences)
            syllable_count = sum(self._count_syllables(word) for word in words)
            avg_syllables_per_word = syllable_count / len(words) if words else 0
            
            # Simplified Flesch Reading Ease formula
            score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            
            # Normalize to 0-100 range
            return max(0, min(100, score))
            
        except Exception:
            return 50.0  # Return neutral score on error
    
    def _count_syllables(self, word: str) -> int:
        """
        Simple syllable counting
        """
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)  # At least one syllable
    
    async def extract_aspects(self, text: str) -> Dict[str, List[str]]:
        """
        Extract product aspects mentioned in the text
        """
        loop = asyncio.get_event_loop()
        
        def _extract():
            text_lower = text.lower()
            extracted_aspects = {}
            
            for aspect, keywords in self.product_aspects.items():
                mentioned_keywords = []
                for keyword in keywords:
                    if keyword in text_lower:
                        # Find the context around the keyword
                        pattern = rf'\b\w*{re.escape(keyword)}\w*\b'
                        matches = re.finditer(pattern, text_lower)
                        for match in matches:
                            mentioned_keywords.append(match.group())
                
                if mentioned_keywords:
                    extracted_aspects[aspect] = list(set(mentioned_keywords))
            
            return extracted_aspects
        
        return await loop.run_in_executor(self.executor, _extract)
    
    async def extract_keywords(self, text: str, top_k: int = 20) -> Dict[str, List[str]]:
        """
        Extract positive and negative keywords
        """
        loop = asyncio.get_event_loop()
        
        def _extract():
            # Preprocess text
            tokens = word_tokenize(text.lower())
            tokens = [word for word in tokens if word.isalpha() and word not in self.stop_words]
            
            # Find sentiment keywords
            positive_found = [word for word in tokens if word in self.positive_words]
            negative_found = [word for word in tokens if word in self.negative_words]
            
            # Get most frequent words (excluding sentiment words)
            from collections import Counter
            neutral_words = [word for word in tokens 
                           if word not in self.positive_words and word not in self.negative_words]
            
            word_freq = Counter(neutral_words)
            top_neutral = [word for word, _ in word_freq.most_common(top_k)]
            
            return {
                'positive': list(set(positive_found)),
                'negative': list(set(negative_found)),
                'neutral': top_neutral
            }
        
        return await loop.run_in_executor(self.executor, _extract)
    
    async def extract_themes(self, text: str, min_frequency: int = 2) -> List[str]:
        """
        Extract main themes from text using NLP techniques
        """
        loop = asyncio.get_event_loop()
        
        def _extract():
            # Tokenize and tag parts of speech
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            # Extract nouns and noun phrases (potential themes)
            nouns = [word.lower() for word, pos in pos_tags 
                    if pos.startswith('NN') and word.isalpha() and len(word) > 2]
            
            # Extract noun phrases
            noun_phrases = []
            i = 0
            while i < len(pos_tags) - 1:
                if pos_tags[i][1].startswith('NN') and pos_tags[i+1][1].startswith('NN'):
                    phrase = f"{pos_tags[i][0]} {pos_tags[i+1][0]}".lower()
                    noun_phrases.append(phrase)
                    i += 2
                else:
                    i += 1
            
            # Count frequency
            from collections import Counter
            all_themes = nouns + noun_phrases
            theme_counts = Counter(all_themes)
            
            # Filter by minimum frequency and relevance
            themes = [theme for theme, count in theme_counts.items() 
                     if count >= min_frequency and len(theme) > 2]
            
            # Sort by frequency
            themes.sort(key=lambda x: theme_counts[x], reverse=True)
            
            return themes[:15]  # Return top 15 themes
        
        return await loop.run_in_executor(self.executor, _extract)
    
    async def extract_pros_and_cons(self, text: str) -> Dict[str, List[str]]:
        """
        Extract pros and cons from review text
        """
        loop = asyncio.get_event_loop()
        
        def _extract():
            sentences = sent_tokenize(text)
            pros = []
            cons = []
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                
                # Check for explicit pros/cons indicators
                if any(indicator in sentence_lower for indicator in ['pros:', 'advantages:', 'good:', 'likes:']):
                    pros.append(sentence.strip())
                    continue
                    
                if any(indicator in sentence_lower for indicator in ['cons:', 'disadvantages:', 'bad:', 'dislikes:']):
                    cons.append(sentence.strip())
                    continue
                
                # Sentiment-based classification
                positive_score = sum(1 for word in self.positive_words 
                                   if word in sentence_lower)
                negative_score = sum(1 for word in self.negative_words 
                                   if word in sentence_lower)
                
                if positive_score > negative_score and positive_score > 0:
                    pros.append(sentence.strip())
                elif negative_score > positive_score and negative_score > 0:
                    cons.append(sentence.strip())
            
            return {
                'pros': pros[:10],  # Limit to top 10
                'cons': cons[:10]
            }
        
        return await loop.run_in_executor(self.executor, _extract)
    
    async def calculate_text_quality_score(self, text: str) -> float:
        """
        Calculate overall text quality score (0-1)
        """
        if not text or len(text.strip()) < 10:
            return 0.0
        
        try:
            # Length score (optimal around 100-500 characters)
            length = len(text)
            if length < 50:
                length_score = length / 50
            elif length > 1000:
                length_score = max(0.5, 1000 / length)
            else:
                length_score = 1.0
            
            # Word diversity score
            words = text.split()
            unique_words = set(word.lower() for word in words)
            diversity_score = len(unique_words) / len(words) if words else 0
            
            # Grammar indicators (simplified)
            punctuation_ratio = sum(1 for char in text if char in string.punctuation) / len(text)
            grammar_score = min(1.0, punctuation_ratio * 10)  # Basic punctuation usage
            
            # Combine scores
            overall_score = (
                length_score * 0.4 +
                diversity_score * 0.4 +
                grammar_score * 0.2
            )
            
            return min(1.0, max(0.0, overall_score))
            
        except Exception:
            return 0.5  # Return neutral score on error