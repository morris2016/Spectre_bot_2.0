import React, { useState, useRef } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  Button, 
  List, 
  ListItem, 
  ListItemText, 
  ListItemIcon, 
  ListItemSecondaryAction, 
  IconButton, 
  Divider, 
  CircularProgress, 
  LinearProgress,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tooltip,
  Alert,
  Snackbar
} from '@mui/material';
import { 
  CloudUpload, 
  Description, 
  Delete, 
  CheckCircle, 
  Error, 
  Info,
  PictureAsPdf,
  InsertDriveFile,
  Article,
  Code,
  Book,
  School,
  MoreVert,
  Edit,
  Visibility
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import api from '../../api';

const DocumentUploader = () => {
  const theme = useTheme();
  const fileInputRef = useRef(null);
  
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({});
  const [processingFiles, setProcessingFiles] = useState([]);
  const [completedFiles, setCompletedFiles] = useState([]);
  const [failedFiles, setFailedFiles] = useState([]);
  const [openDialog, setOpenDialog] = useState(false);
  const [currentFile, setCurrentFile] = useState(null);
  const [documentType, setDocumentType] = useState('trading_strategy');
  const [documentDescription, setDocumentDescription] = useState('');
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'info'
  });

  const handleFileSelect = (event) => {
    const selectedFiles = Array.from(event.target.files);
    setFiles(prev => [...prev, ...selectedFiles]);
    
    // Reset file input
    event.target.value = null;
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const handleDrop = (event) => {
    event.preventDefault();
    
    if (event.dataTransfer.files && event.dataTransfer.files.length > 0) {
      const droppedFiles = Array.from(event.dataTransfer.files);
      setFiles(prev => [...prev, ...droppedFiles]);
    }
  };

  const handleRemoveFile = (index) => {
    const newFiles = [...files];
    newFiles.splice(index, 1);
    setFiles(newFiles);
  };

  const handleUpload = async () => {
    if (files.length === 0) return;
    
    setUploading(true);
    const newUploadProgress = {};
    files.forEach(file => {
      newUploadProgress[file.name] = 0;
    });
    setUploadProgress(newUploadProgress);
    
    for (const file of files) {
      try {
        setProcessingFiles(prev => [...prev, file.name]);
        
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        formData.append('type', documentType);
        formData.append('description', documentDescription);
        
        // Upload file with progress tracking
        const response = await api.post('/api/learning/upload-document', formData, {
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setUploadProgress(prev => ({
              ...prev,
              [file.name]: percentCompleted
            }));
          }
        });
        
        if (response.data && response.data.success) {
          setCompletedFiles(prev => [...prev, {
            name: file.name,
            id: response.data.document_id,
            type: documentType,
            insights: response.data.insights || [],
            status: 'completed'
          }]);
          
          setSnackbar({
            open: true,
            message: `Successfully processed ${file.name}`,
            severity: 'success'
          });
        } else {
          throw new Error(response.data?.message || 'Unknown error');
        }
      } catch (err) {
        console.error(`Error uploading ${file.name}:`, err);
        
        setFailedFiles(prev => [...prev, {
          name: file.name,
          error: err.message || 'Upload failed'
        }]);
        
        setSnackbar({
          open: true,
          message: `Failed to process ${file.name}: ${err.message}`,
          severity: 'error'
        });
      } finally {
        setProcessingFiles(prev => prev.filter(name => name !== file.name));
      }
    }
    
    setUploading(false);
    setFiles([]);
    setDocumentDescription('');
  };

  const handleOpenDialog = () => {
    if (files.length === 0) {
      setSnackbar({
        open: true,
        message: 'Please select files to upload first',
        severity: 'warning'
      });
      return;
    }
    
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
  };

  const handleDocumentTypeChange = (event) => {
    setDocumentType(event.target.value);
  };

  const handleDescriptionChange = (event) => {
    setDocumentDescription(event.target.value);
  };

  const handleViewDocument = (document) => {
    // This would open a dialog to view document insights
    setCurrentFile(document);
    // Additional logic to fetch document details would go here
  };

  const handleCloseSnackbar = () => {
    setSnackbar(prev => ({
      ...prev,
      open: false
    }));
  };

  const getFileIcon = (fileName) => {
    const extension = fileName.split('.').pop().toLowerCase();
    
    switch (extension) {
      case 'pdf':
        return <PictureAsPdf color="error" />;
      case 'doc':
      case 'docx':
        return <Description color="primary" />;
      case 'txt':
        return <Article color="action" />;
      case 'py':
      case 'js':
      case 'java':
      case 'cpp':
      case 'c':
      case 'h':
      case 'cs':
        return <Code color="secondary" />;
      default:
        return <InsertDriveFile />;
    }
  };

  const getDocumentTypeIcon = (type) => {
    switch (type) {
      case 'trading_strategy':
        return <Article />;
      case 'market_analysis':
        return <BarChart />;
      case 'research_paper':
        return <School />;
      case 'book':
        return <Book />;
      default:
        return <Description />;
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <Paper elevation={0} variant="outlined">
      <Box p={2}>
        <Typography variant="h6" gutterBottom>Document Learning</Typography>
        <Typography variant="body2" color="textSecondary" paragraph>
          Upload documents for the AI to learn new trading strategies and market analysis techniques.
        </Typography>
        
        <Box 
          border={2} 
          borderRadius={1} 
          borderColor="divider" 
          borderStyle="dashed" 
          p={3} 
          textAlign="center"
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          bgcolor={theme.palette.background.default}
          mb={3}
        >
          <input
            type="file"
            multiple
            onChange={handleFileSelect}
            style={{ display: 'none' }}
            ref={fileInputRef}
            accept=".pdf,.doc,.docx,.txt,.md,.py,.js,.ipynb"
          />
          
          <CloudUpload style={{ fontSize: 48, color: theme.palette.primary.main, marginBottom: theme.spacing(2) }} />
          
          <Typography variant="body1" gutterBottom>
            Drag and drop files here, or
          </Typography>
          
          <Button 
            variant="contained" 
            color="primary" 
            onClick={() => fileInputRef.current.click()}
            disabled={uploading}
          >
            Browse Files
          </Button>
          
          <Typography variant="caption" display="block" color="textSecondary" mt={1}>
            Supported formats: PDF, DOC, DOCX, TXT, MD, PY, JS, IPYNB
          </Typography>
        </Box>
        
        {files.length > 0 && (
          <Box mb={3}>
            <Typography variant="subtitle2" gutterBottom>
              Selected Files ({files.length})
            </Typography>
            
            <List>
              {files.map((file, index) => (
                <ListItem key={index}>
                  <ListItemIcon>
                    {getFileIcon(file.name)}
                  </ListItemIcon>
                  <ListItemText 
                    primary={file.name} 
                    secondary={formatFileSize(file.size)} 
                  />
                  <ListItemSecondaryAction>
                    <IconButton 
                      edge="end" 
                      onClick={() => handleRemoveFile(index)}
                      disabled={uploading}
                    >
                      <Delete />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
            
            <Box mt={2} display="flex" justifyContent="flex-end">
              <Button 
                variant="outlined" 
                color="secondary" 
                onClick={() => setFiles([])}
                disabled={uploading}
                style={{ marginRight: theme.spacing(1) }}
              >
                Clear All
              </Button>
              <Button 
                variant="contained" 
                color="primary" 
                onClick={handleOpenDialog}
                disabled={uploading}
              >
                Continue
              </Button>
            </Box>
          </Box>
        )}
        
        {processingFiles.length > 0 && (
          <Box mb={3}>
            <Typography variant="subtitle2" gutterBottom>
              Processing Files
            </Typography>
            
            <List>
              {processingFiles.map((fileName, index) => (
                <ListItem key={index}>
                  <ListItemIcon>
                    {getFileIcon(fileName)}
                  </ListItemIcon>
                  <ListItemText 
                    primary={fileName} 
                    secondary={
                      <LinearProgress 
                        variant="determinate" 
                        value={uploadProgress[fileName] || 0} 
                        style={{ marginTop: theme.spacing(1) }}
                      />
                    } 
                  />
                  <ListItemSecondaryAction>
                    <CircularProgress size={24} />
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
          </Box>
        )}
        
        {completedFiles.length > 0 && (
          <Box mb={3}>
            <Typography variant="subtitle2" gutterBottom>
              Processed Documents
            </Typography>
            
            <List>
              {completedFiles.map((file, index) => (
                <ListItem key={index}>
                  <ListItemIcon>
                    {getDocumentTypeIcon(file.type)}
                  </ListItemIcon>
                  <ListItemText 
                    primary={file.name} 
                    secondary={
                      <Box display="flex" alignItems="center" mt={0.5}>
                        <Chip 
                          label={file.type.replace('_', ' ')} 
                          size="small" 
                          style={{ textTransform: 'capitalize' }}
                        />
                        {file.insights && file.insights.length > 0 && (
                          <Chip 
                            icon={<Info />}
                            label={`${file.insights.length} insights`} 
                            size="small"
                            color="primary" 
                            style={{ marginLeft: theme.spacing(1) }}
                          />
                        )}
                      </Box>
                    } 
                  />
                  <ListItemSecondaryAction>
                    <Tooltip title="View Document">
                      <IconButton 
                        edge="end" 
                        onClick={() => handleViewDocument(file)}
                      >
                        <Visibility />
                      </IconButton>
                    </Tooltip>
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
          </Box>
        )}
        
        {failedFiles.length > 0 && (
          <Box mb={3}>
            <Typography variant="subtitle2" color="error" gutterBottom>
              Failed Documents
            </Typography>
            
            <List>
              {failedFiles.map((file, index) => (
                <ListItem key={index}>
                  <ListItemIcon>
                    <Error color="error" />
                  </ListItemIcon>
                  <ListItemText 
                    primary={file.name} 
                    secondary={file.error} 
                  />
                  <ListItemSecondaryAction>
                    <IconButton edge="end" onClick={() => setFailedFiles(prev => prev.filter((_, i) => i !== index))}>
                      <Delete />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
          </Box>
        )}
      </Box>
      
      <Dialog open={openDialog} onClose={handleCloseDialog}>
        <DialogTitle>Document Information</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Please provide additional information about the document(s) you're uploading.
            This helps the AI understand and learn from the content more effectively.
          </DialogContentText>
          
          <FormControl fullWidth margin="normal">
            <InputLabel id="document-type-label">Document Type</InputLabel>
            <Select
              labelId="document-type-label"
              value={documentType}
              onChange={handleDocumentTypeChange}
              label="Document Type"
            >
              <MenuItem value="trading_strategy">Trading Strategy</MenuItem>
              <MenuItem value="market_analysis">Market Analysis</MenuItem>
              <MenuItem value="research_paper">Research Paper</MenuItem>
              <MenuItem value="book">Book</MenuItem>
              <MenuItem value="code_example">Code Example</MenuItem>
              <MenuItem value="other">Other</MenuItem>
            </Select>
          </FormControl>
          
          <TextField
            margin="normal"
            label="Description (optional)"
            fullWidth
            multiline
            rows={4}
            value={documentDescription}
            onChange={handleDescriptionChange}
            placeholder="Briefly describe what this document contains and what you want the AI to learn from it."
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog} color="primary">
            Cancel
          </Button>
          <Button 
            onClick={() => {
              handleCloseDialog();
              handleUpload();
            }} 
            color="primary" 
            variant="contained"
          >
            Upload and Process
          </Button>
        </DialogActions>
      </Dialog>
      
      <Snackbar 
        open={snackbar.open} 
        autoHideDuration={6000} 
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert 
          onClose={handleCloseSnackbar} 
          severity={snackbar.severity} 
          variant="filled"
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Paper>
  );
};

export default DocumentUploader;