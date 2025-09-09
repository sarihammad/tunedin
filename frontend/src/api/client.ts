import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('API Response Error:', error);
    return Promise.reject(error);
  }
);

export interface RecommendationItem {
  track_id: string;
  score: number;
}

export interface RecommendationResponse {
  user_id: string;
  items: RecommendationItem[];
  total: number;
  cache_hit: boolean;
}

export interface FeedbackRequest {
  user_id: string;
  track_id: string;
  event: 'play' | 'like' | 'skip';
  ts: number;
}

export interface FeedbackResponse {
  success: boolean;
  message: string;
}

// API functions
export const recommendationsApi = {
  getUserRecommendations: async (userId: string, n: number = 50): Promise<RecommendationResponse> => {
    const response = await apiClient.get(`/rec/users/${userId}?n=${n}`);
    return response.data;
  },

  getSimilarUsers: async (userId: string, n: number = 20): Promise<RecommendationResponse> => {
    const response = await apiClient.get(`/rec/users/${userId}/similar?n=${n}`);
    return response.data;
  },
};

export const feedbackApi = {
  submitFeedback: async (feedback: FeedbackRequest): Promise<FeedbackResponse> => {
    const response = await apiClient.post('/feedback/', feedback);
    return response.data;
  },

  getFeedbackStats: async () => {
    const response = await apiClient.get('/feedback/stats');
    return response.data;
  },
};

export const healthApi = {
  checkHealth: async () => {
    const response = await apiClient.get('/healthz');
    return response.data;
  },

  getStatus: async () => {
    const response = await apiClient.get('/status');
    return response.data;
  },
};

