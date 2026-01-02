import React from 'react';
import { Container, Typography, Alert } from '@mui/material';

export const RegisterPage: React.FC = () => {
  return (
    <Container maxWidth="sm" sx={{ py: 4 }}>
      <Alert severity="info" sx={{ mb: 4 }}>
        Authentication pages will be fully implemented once backend auth is tested
      </Alert>
      <Typography variant="h4">Register</Typography>
    </Container>
  );
};