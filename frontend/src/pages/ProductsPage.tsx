import React from 'react';
import {
  Box,
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  CardMedia,
  Button,
  Chip,
  TextField,
  InputAdornment,
  Alert,
} from '@mui/material';
import { Search, ShoppingCart } from '@mui/icons-material';

// Mock product data for Phase 1
const mockProducts = [
  {
    id: '1',
    title: 'iPhone 15 Pro',
    price: 999.99,
    original_price: 1099.99,
    image: '/api/placeholder/300/200',
    rating: 4.8,
    reviews: 245,
    category: 'Electronics',
    is_featured: true,
  },
  {
    id: '2',
    title: 'Samsung Galaxy S24 Ultra',
    price: 1199.99,
    image: '/api/placeholder/300/200',
    rating: 4.7,
    reviews: 189,
    category: 'Electronics',
    is_featured: false,
  },
  // Add more mock products...
];

export const ProductsPage: React.FC = () => {
  const [searchQuery, setSearchQuery] = React.useState('');

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom>
          Products
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
          Discover our curated collection of premium products
        </Typography>

        {/* Search Bar */}
        <TextField
          fullWidth
          placeholder="Search products..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search />
              </InputAdornment>
            ),
          }}
          sx={{ maxWidth: 600 }}
        />
      </Box>

      {/* Phase 1 Notice */}
      <Alert severity="info" sx={{ mb: 4 }}>
        <Typography variant="body2">
          <strong>Phase 1 Preview:</strong> This is a basic product catalog. 
          Advanced features like AI-powered recommendations, semantic search, and review analysis 
          will be implemented in subsequent phases.
        </Typography>
      </Alert>

      {/* Products Grid */}
      <Grid container spacing={3}>
        {mockProducts.map((product) => (
          <Grid item xs={12} sm={6} md={4} lg={3} key={product.id}>
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
              <CardMedia
                component="div"
                sx={{
                  height: 200,
                  backgroundColor: 'grey.200',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  position: 'relative',
                }}
              >
                {product.is_featured && (
                  <Chip
                    label="Featured"
                    color="secondary"
                    size="small"
                    sx={{
                      position: 'absolute',
                      top: 8,
                      left: 8,
                    }}
                  />
                )}
                <Typography color="text.secondary">
                  Product Image
                </Typography>
              </CardMedia>

              <CardContent sx={{ flexGrow: 1 }}>
                <Typography variant="h6" component="h2" gutterBottom noWrap>
                  {product.title}
                </Typography>
                
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Typography variant="h6" color="primary" sx={{ mr: 1 }}>
                    ${product.price}
                  </Typography>
                  {product.original_price && (
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{ textDecoration: 'line-through' }}
                    >
                      ${product.original_price}
                    </Typography>
                  )}
                </Box>

                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    ⭐ {product.rating} ({product.reviews} reviews)
                  </Typography>
                </Box>

                <Chip
                  label={product.category}
                  size="small"
                  variant="outlined"
                  sx={{ mb: 2 }}
                />
              </CardContent>

              <Box sx={{ p: 2, pt: 0 }}>
                <Button
                  variant="contained"
                  fullWidth
                  startIcon={<ShoppingCart />}
                  disabled
                >
                  Add to Cart (Soon)
                </Button>
              </Box>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Empty State */}
      {mockProducts.length === 0 && (
        <Box sx={{ textAlign: 'center', py: 8 }}>
          <Typography variant="h5" color="text.secondary" gutterBottom>
            No products found
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Try adjusting your search criteria
          </Typography>
        </Box>
      )}

      {/* Coming Soon Features */}
      <Box sx={{ mt: 6, p: 3, bgcolor: 'background.default', borderRadius: 2 }}>
        <Typography variant="h5" gutterBottom>
          Coming Soon in Future Phases
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <Typography variant="body2" color="text.secondary">
              • AI-powered product recommendations<br />
              • Semantic search with natural language<br />
              • Advanced filtering and sorting<br />
              • Price comparison and alerts
            </Typography>
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography variant="body2" color="text.secondary">
              • Review analysis and sentiment scores<br />
              • Similar product suggestions<br />
              • Personalized product rankings<br />
              • Real-time inventory updates
            </Typography>
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
};