import React from 'react';
import {
  Box,
  Container,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  CardActions,
  Chip,
} from '@mui/material';
import {
  Psychology,
  TrendingUp,
  ChatBubble,
  Description,
  Analytics,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

const features = [
  {
    icon: <Psychology />,
    title: 'Product Recommendations',
    description: 'RAG-based personalized product recommendations using vector embeddings and semantic search.',
    status: 'Phase 2',
    color: 'primary' as const,
  },
  {
    icon: <Analytics />,
    title: 'Review Analysis',
    description: 'Intelligent sentiment analysis and review summarization with theme extraction.',
    status: 'Phase 3',
    color: 'secondary' as const,
  },
  {
    icon: <ChatBubble />,
    title: 'AI Chatbot',
    description: 'Conversational AI for customer support with RAG-enhanced responses.',
    status: 'Phase 4',
    color: 'success' as const,
  },
  {
    icon: <Description />,
    title: 'Content Generation',
    description: 'Automated SEO-optimized product descriptions and marketing copy.',
    status: 'Phase 5',
    color: 'warning' as const,
  },
  {
    icon: <TrendingUp />,
    title: 'Business Analytics',
    description: 'Comprehensive business intelligence and performance insights.',
    status: 'Phase 6',
    color: 'info' as const,
  },
];

export const HomePage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <Box>
      {/* Hero Section */}
      <Box
        sx={{
          background: 'linear-gradient(135deg, #1976d2 0%, #42a5f5 100%)',
          color: 'white',
          py: 8,
          mb: 6,
        }}
      >
        <Container maxWidth="lg">
          <Grid container spacing={4} alignItems="center">
            <Grid item xs={12} md={8}>
              <Typography variant="h2" component="h1" gutterBottom>
                EcommerceAgents
              </Typography>
              <Typography variant="h5" component="h2" gutterBottom sx={{ opacity: 0.9 }}>
                Multi-Agent E-Commerce Intelligence System
              </Typography>
              <Typography variant="body1" sx={{ mb: 4, opacity: 0.8, fontSize: '1.1rem' }}>
                Leverage the power of AI agents with RAG (Retrieval-Augmented Generation) for 
                intelligent product recommendations, review analysis, conversational chatbots, 
                and automated content generation.
              </Typography>
              <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                <Button
                  variant="contained"
                  color="secondary"
                  size="large"
                  onClick={() => navigate('/products')}
                >
                  Explore Products
                </Button>
                <Button
                  variant="outlined"
                  color="inherit"
                  size="large"
                  onClick={() => navigate('/login')}
                >
                  Get Started
                </Button>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: 'center' }}>
                <Box
                  sx={{
                    width: 200,
                    height: 200,
                    border: '3px solid rgba(255,255,255,0.3)',
                    borderRadius: '50%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    margin: '0 auto',
                    backgroundColor: 'rgba(255,255,255,0.1)',
                  }}
                >
                  <Psychology sx={{ fontSize: 80 }} />
                </Box>
              </Box>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Features Section */}
      <Container maxWidth="lg" sx={{ mb: 8 }}>
        <Typography variant="h3" component="h2" textAlign="center" gutterBottom>
          AI-Powered E-Commerce Features
        </Typography>
        <Typography variant="body1" textAlign="center" color="text.secondary" sx={{ mb: 6 }}>
          Our multi-agent system provides comprehensive e-commerce intelligence across all phases of development
        </Typography>

        <Grid container spacing={4}>
          {features.map((feature, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              <Card
                sx={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  transition: 'transform 0.2s',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                  },
                }}
              >
                <CardContent sx={{ flexGrow: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Box sx={{ color: 'primary.main', mr: 2 }}>
                      {feature.icon}
                    </Box>
                    <Typography variant="h6" component="h3">
                      {feature.title}
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    {feature.description}
                  </Typography>
                  <Chip
                    label={feature.status}
                    color={feature.color}
                    size="small"
                    variant="outlined"
                  />
                </CardContent>
                <CardActions>
                  <Button size="small" disabled>
                    Coming Soon
                  </Button>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Container>

      {/* Technology Stack Section */}
      <Box sx={{ bgcolor: 'background.default', py: 8 }}>
        <Container maxWidth="lg">
          <Typography variant="h3" component="h2" textAlign="center" gutterBottom>
            Technology Stack
          </Typography>
          <Grid container spacing={4} sx={{ mt: 2 }}>
            <Grid item xs={12} md={6}>
              <Typography variant="h5" gutterBottom>
                Backend & AI/ML
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                • Python with FastAPI for high-performance APIs<br />
                • PostgreSQL + Vector Database (Pinecone/Weaviate)<br />
                • OpenAI GPT-4, Claude, Sentence-Transformers<br />
                • Redis/Celery for agent communication<br />
                • Comprehensive analytics and monitoring
              </Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="h5" gutterBottom>
                Frontend & Infrastructure
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                • React.js with TypeScript and Material-UI<br />
                • Real-time WebSocket connections<br />
                • Docker containerization<br />
                • GitHub Actions CI/CD pipeline<br />
                • Prometheus monitoring and metrics
              </Typography>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* CTA Section */}
      <Container maxWidth="lg" sx={{ py: 8, textAlign: 'center' }}>
        <Typography variant="h4" component="h2" gutterBottom>
          Ready to Get Started?
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
          Join the future of e-commerce with AI-powered intelligence
        </Typography>
        <Button
          variant="contained"
          size="large"
          onClick={() => navigate('/register')}
        >
          Start Your Journey
        </Button>
      </Container>
    </Box>
  );
};