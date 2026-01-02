// User types
export interface User {
  id: string;
  email: string;
  username: string;
  full_name?: string;
  is_active: boolean;
  preferences?: Record<string, any>;
  created_at: string;
  last_login?: string;
}

export interface AuthToken {
  access_token: string;
  token_type: string;
  expires_in: number;
}

// Product types
export interface Product {
  id: string;
  title: string;
  description?: string;
  short_description?: string;
  sku: string;
  price: number;
  original_price?: number;
  currency: string;
  stock_quantity: number;
  is_in_stock: boolean;
  attributes?: Record<string, any>;
  features?: string[];
  tags?: string[];
  images?: string[];
  primary_image?: string;
  slug: string;
  average_rating: number;
  review_count: number;
  is_featured: boolean;
  category?: Category;
  brand?: Brand;
  created_at: string;
}

export interface Category {
  id: string;
  name: string;
  description?: string;
  slug: string;
  image_url?: string;
  level: number;
  path?: string;
}

export interface Brand {
  id: string;
  name: string;
  description?: string;
  logo_url?: string;
  website_url?: string;
}

// Search types
export interface SearchQuery {
  query: string;
  filters?: Record<string, any>;
  limit?: number;
  offset?: number;
  sort_by?: string;
  sort_order?: string;
}

export interface SearchResult {
  products: Product[];
  total_results: number;
  response_time: number;
  algorithm_used: string;
}

// Recommendation types
export interface RecommendationRequest {
  user_id?: string;
  query: string;
  context?: string;
  filters?: Record<string, any>;
  limit?: number;
  include_metadata?: boolean;
}

export interface RecommendationResponse {
  products: Product[];
  query: string;
  context: string;
  total_results: number;
  response_time: number;
  algorithm_used: string;
  metadata?: Record<string, any>;
}

// Analytics types
export interface AnalyticsData {
  total_users: number;
  total_products: number;
  total_interactions: number;
  conversion_rate: number;
  revenue: number;
  top_categories: Array<Record<string, any>>;
  user_growth: Array<Record<string, any>>;
}

// API response types
export interface ApiResponse<T> {
  data: T;
  message?: string;
  status: number;
}

export interface ApiError {
  detail: string;
  status?: number;
}