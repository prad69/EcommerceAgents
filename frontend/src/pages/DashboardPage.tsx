import React from 'react';
import { Container, Typography, Alert } from '@mui/material';

export const DashboardPage: React.FC = () => {
  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Alert severity="info" sx={{ mb: 4 }}>
        User dashboard will be implemented in Phase 2
      </Alert>
      <Typography variant="h4">Dashboard</Typography>
    </Container>
  );
};