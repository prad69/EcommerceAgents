import React from 'react';
import { Container, Typography, Alert } from '@mui/material';

export const AnalyticsPage: React.FC = () => {
  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Alert severity="info" sx={{ mb: 4 }}>
        Analytics dashboard will be implemented in Phase 6
      </Alert>
      <Typography variant="h4">Analytics</Typography>
    </Container>
  );
};