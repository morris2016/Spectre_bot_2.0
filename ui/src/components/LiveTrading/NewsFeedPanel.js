import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  CircularProgress, 
  List, 
  ListItem, 
  ListItemText, 
  ListItemIcon, 
  Divider, 
  Chip, 
  IconButton, 
  Tooltip,
  Button,
  Menu,
  MenuItem,
  FormControl,
  InputLabel,
  Select,
  TextField,
  InputAdornment
} from '@mui/material';
import { 
  Refresh, 
  TrendingUp, 
  TrendingDown, 
  TrendingFlat, 
  Article, 
  FilterList,
  Search,
  OpenInNew,
  Bookmark,
  BookmarkBorder,
  ThumbUp,
  ThumbDown,
  MoreVert
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import api from '../../api';

const NewsFeedPanel = ({ 
  asset = null, 
  maxItems = 10,
  showSentiment = true,
  height = 400
}) => {
  const theme = useTheme();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [news, setNews] = useState([]);
  const [filteredNews, setFilteredNews] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [sentimentFilter, setSentimentFilter] = useState('all');
  const [sourceFilter, setSourceFilter] = useState('all');
  const [sources, setSources] = useState([]);
  const [anchorEl, setAnchorEl] = useState(null);
  const [selectedNewsItem, setSelectedNewsItem] = useState(null);
  const [savedArticles, setSavedArticles] = useState([]);

  // Fetch news data
  useEffect(() => {
    fetchNews();
  }, [asset]);

  // Filter news when filters change
  useEffect(() => {
    filterNews();
  }, [news, searchTerm, sentimentFilter, sourceFilter]);

  const fetchNews = async () => {
    setLoading(true);
    setError(null);

    try {
      let endpoint = '/api/market/news';
      if (asset) {
        endpoint += `?asset=${asset}`;
      }

      const response = await api.get(endpoint);
      
      if (response.data && response.data.news) {
        setNews(response.data.news);
        
        // Extract unique sources
        const uniqueSources = [...new Set(response.data.news.map(item => item.source))];
        setSources(uniqueSources);
      }
    } catch (err) {
      console.error('Error fetching news:', err);
      setError('Failed to load news data');
    } finally {
      setLoading(false);
    }
  };

  const filterNews = () => {
    let filtered = [...news];
    
    // Apply search filter
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(item => 
        item.title.toLowerCase().includes(term) || 
        item.summary.toLowerCase().includes(term)
      );
    }
    
    // Apply sentiment filter
    if (sentimentFilter !== 'all') {
      filtered = filtered.filter(item => item.sentiment === sentimentFilter);
    }
    
    // Apply source filter
    if (sourceFilter !== 'all') {
      filtered = filtered.filter(item => item.source === sourceFilter);
    }
    
    // Limit number of items
    filtered = filtered.slice(0, maxItems);
    
    setFilteredNews(filtered);
  };

  const handleSearchChange = (event) => {
    setSearchTerm(event.target.value);
  };

  const handleSentimentFilterChange = (event) => {
    setSentimentFilter(event.target.value);
  };

  const handleSourceFilterChange = (event) => {
    setSourceFilter(event.target.value);
  };

  const handleMenuOpen = (event, item) => {
    setAnchorEl(event.currentTarget);
    setSelectedNewsItem(item);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
    setSelectedNewsItem(null);
  };

  const handleSaveArticle = (item) => {
    const isAlreadySaved = savedArticles.some(article => article.id === item.id);
    
    if (isAlreadySaved) {
      setSavedArticles(savedArticles.filter(article => article.id !== item.id));
    } else {
      setSavedArticles([...savedArticles, item]);
    }
    
    handleMenuClose();
  };

  const handleOpenArticle = (url) => {
    window.open(url, '_blank');
    handleMenuClose();
  };

  const getSentimentIcon = (sentiment) => {
    switch (sentiment) {
      case 'positive':
        return <TrendingUp style={{ color: theme.palette.success.main }} />;
      case 'negative':
        return <TrendingDown style={{ color: theme.palette.error.main }} />;
      default:
        return <TrendingFlat style={{ color: theme.palette.text.secondary }} />;
    }
  };

  const getSentimentColor = (sentiment) => {
    switch (sentiment) {
      case 'positive':
        return theme.palette.success.main;
      case 'negative':
        return theme.palette.error.main;
      default:
        return theme.palette.text.secondary;
    }
  };

  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  if (loading && !news.length) {
    return (
      <Box 
        display="flex" 
        justifyContent="center" 
        alignItems="center" 
        height={height} 
        border={1} 
        borderColor="divider" 
        borderRadius={1}
      >
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box 
        display="flex" 
        justifyContent="center" 
        alignItems="center" 
        height={height} 
        border={1} 
        borderColor="divider" 
        borderRadius={1}
      >
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }

  return (
    <Paper elevation={0} variant="outlined">
      <Box p={2} display="flex" justifyContent="space-between" alignItems="center">
        <Typography variant="h6">Market News</Typography>
        
        <Box display="flex" alignItems="center">
          <Tooltip title="Refresh">
            <IconButton onClick={fetchNews} size="small">
              <Refresh />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
      
      <Divider />
      
      <Box p={2} display="flex" alignItems="center" flexWrap="wrap" gap={2}>
        <TextField
          placeholder="Search news..."
          variant="outlined"
          size="small"
          value={searchTerm}
          onChange={handleSearchChange}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search />
              </InputAdornment>
            ),
          }}
          style={{ flexGrow: 1, minWidth: 200 }}
        />
        
        <FormControl size="small" variant="outlined" style={{ minWidth: 120 }}>
          <InputLabel id="sentiment-filter-label">Sentiment</InputLabel>
          <Select
            labelId="sentiment-filter-label"
            value={sentimentFilter}
            onChange={handleSentimentFilterChange}
            label="Sentiment"
          >
            <MenuItem value="all">All</MenuItem>
            <MenuItem value="positive">Positive</MenuItem>
            <MenuItem value="neutral">Neutral</MenuItem>
            <MenuItem value="negative">Negative</MenuItem>
          </Select>
        </FormControl>
        
        <FormControl size="small" variant="outlined" style={{ minWidth: 120 }}>
          <InputLabel id="source-filter-label">Source</InputLabel>
          <Select
            labelId="source-filter-label"
            value={sourceFilter}
            onChange={handleSourceFilterChange}
            label="Source"
          >
            <MenuItem value="all">All Sources</MenuItem>
            {sources.map((source) => (
              <MenuItem key={source} value={source}>{source}</MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>
      
      <Box 
        sx={{ 
          maxHeight: height, 
          overflowY: 'auto',
          '&::-webkit-scrollbar': {
            width: '8px',
          },
          '&::-webkit-scrollbar-thumb': {
            backgroundColor: theme.palette.divider,
            borderRadius: '4px',
          }
        }}
      >
        <List disablePadding>
          {filteredNews.length > 0 ? (
            filteredNews.map((item, index) => (
              <React.Fragment key={item.id || index}>
                <ListItem 
                  alignItems="flex-start"
                  secondaryAction={
                    <IconButton 
                      edge="end" 
                      aria-label="more"
                      onClick={(e) => handleMenuOpen(e, item)}
                    >
                      <MoreVert />
                    </IconButton>
                  }
                >
                  <ListItemIcon>
                    <Article />
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Box display="flex" alignItems="center">
                        <Typography variant="subtitle2" style={{ flexGrow: 1 }}>
                          {item.title}
                        </Typography>
                        {showSentiment && (
                          <Chip 
                            icon={getSentimentIcon(item.sentiment)} 
                            label={item.sentiment}
                            size="small"
                            style={{ 
                              backgroundColor: getSentimentColor(item.sentiment),
                              color: theme.palette.getContrastText(getSentimentColor(item.sentiment)),
                              textTransform: 'capitalize',
                              marginLeft: theme.spacing(1)
                            }}
                          />
                        )}
                      </Box>
                    }
                    secondary={
                      <>
                        <Typography
                          variant="body2"
                          color="textSecondary"
                          component="span"
                        >
                          {item.summary}
                        </Typography>
                        <Box mt={1} display="flex" justifyContent="space-between" alignItems="center">
                          <Typography variant="caption" color="textSecondary">
                            {item.source} • {formatDate(item.timestamp)}
                          </Typography>
                          {savedArticles.some(article => article.id === item.id) && (
                            <Bookmark fontSize="small" color="primary" />
                          )}
                        </Box>
                      </>
                    }
                  />
                </ListItem>
                {index < filteredNews.length - 1 && <Divider variant="inset" component="li" />}
              </React.Fragment>
            ))
          ) : (
            <ListItem>
              <ListItemText
                primary="No news found"
                secondary="Try changing your filters or search term"
              />
            </ListItem>
          )}
        </List>
      </Box>
      
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => selectedNewsItem && handleOpenArticle(selectedNewsItem.url)}>
          <ListItemIcon>
            <OpenInNew fontSize="small" />
          </ListItemIcon>
          <ListItemText primary="Open Article" />
        </MenuItem>
        <MenuItem onClick={() => selectedNewsItem && handleSaveArticle(selectedNewsItem)}>
          <ListItemIcon>
            {savedArticles.some(article => article.id === selectedNewsItem?.id) ? (
              <Bookmark fontSize="small" />
            ) : (
              <BookmarkBorder fontSize="small" />
            )}
          </ListItemIcon>
          <ListItemText 
            primary={savedArticles.some(article => article.id === selectedNewsItem?.id) ? 
              "Remove from Saved" : "Save Article"} 
          />
        </MenuItem>
        <Divider />
        <MenuItem>
          <ListItemIcon>
            <ThumbUp fontSize="small" />
          </ListItemIcon>
          <ListItemText primary="Relevant" />
        </MenuItem>
        <MenuItem>
          <ListItemIcon>
            <ThumbDown fontSize="small" />
          </ListItemIcon>
          <ListItemText primary="Not Relevant" />
        </MenuItem>
      </Menu>
    </Paper>
  );
};

export default NewsFeedPanel;