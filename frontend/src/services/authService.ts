import axios from 'axios';
import { User, AuthToken } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const authService = {
  async login(email: string, password: string): Promise<AuthToken> {
    const response = await api.post('/auth/login', {
      email,
      password,
    });
    return response.data;
  },

  async register(userData: {
    email: string;
    username: string;
    password: string;
    full_name?: string;
    preferences?: Record<string, any>;
  }): Promise<User> {
    const response = await api.post('/auth/register', userData);
    return response.data;
  },

  async logout(token: string): Promise<void> {
    await api.post('/auth/logout', {}, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
  },

  async getCurrentUser(token: string): Promise<User> {
    const response = await api.get('/auth/me', {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    return response.data;
  },

  async refreshToken(token: string): Promise<AuthToken> {
    const response = await api.post('/auth/refresh', {}, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    return response.data;
  },
};