import axios from 'axios';

const API_BASE_URL = '/api/audit';

export const evaluateAIDecision = async (formData) => {
  const response = await axios.post(`${API_BASE_URL}/evaluate`, formData);
  return response.data;
};

export const submitHumanReview = async (transactionId, reviewerDecision) => {
  const response = await axios.post(`${API_BASE_URL}/review`, {
    transactionId,
    reviewerDecision
  });
  return response.data;
};

export const fetchAuditRecords = async () => {
  const response = await axios.get(`${API_BASE_URL}/records`);
  return response.data;
};

export const verifyAuditRecord = async (transactionId) => {
  const response = await axios.get(`${API_BASE_URL}/${transactionId}/verify`);
  return response.data;
};

export const simulateTamperRecord = async (transactionId, field = 'riskScore', tamperedValue = 20) => {
  const response = await axios.post(`${API_BASE_URL}/${transactionId}/tamper`, {
    field,
    tamperedValue
  });
  return response.data;
};

export const fetchDashboardStats = async () => {
  const response = await axios.get(`${API_BASE_URL}/stats`);
  return response.data;
};

export const resetDemoData = async () => {
  const response = await axios.post(`${API_BASE_URL}/reset`);
  return response.data;
};
