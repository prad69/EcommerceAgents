import React from 'react';
import { Container, Typography, Alert } from '@mui/material';
import { useParams } from 'react-router-dom';

export const ProductDetailPage: React.FC = () => {
  const { id } = useParams();

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Alert severity="info" sx={{ mb: 4 }}>
        Product detail page will be implemented in Phase 2
      </Alert>
      <Typography variant="h4">Product Detail: {id}</Typography>
    </Container>
  );
};