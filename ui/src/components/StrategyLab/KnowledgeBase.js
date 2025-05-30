import React, { useState } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  Accordion, 
  AccordionSummary, 
  AccordionDetails, 
  List, 
  ListItem, 
  ListItemIcon, 
  ListItemText, 
  Divider, 
  Chip, 
  Button,
  TextField,
  InputAdornment,
  IconButton,
  Grid,
  Card,
  CardContent,
  CardHeader,
  CardActions
} from '@mui/material';
import { 
  ExpandMore, 
  Search, 
  School, 
  Psychology, 
  TrendingUp, 
  ShowChart, 
  BarChart, 
  Timeline, 
  Lightbulb,
  BookmarkBorder,
  Bookmark,
  ArrowForward,
  Info
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';

// Trading knowledge categories
const KNOWLEDGE_CATEGORIES = [
  {
    id: 'price_action',
    name: 'Price Action',
    description: 'Pure price movement analysis without indicators',
    icon: <ShowChart />,
    color: '#2196f3',
    concepts: [
      {
        name: 'Support & Resistance',
        description: 'Price levels where the market has historically reversed direction',
        difficulty: 'beginner',
        resources: [
          { type: 'article', title: 'Identifying Key Support and Resistance Levels' },
          { type: 'video', title: 'How to Trade Support and Resistance Like a Pro' }
        ]
      },
      {
        name: 'Chart Patterns',
        description: 'Recognizable patterns that form on price charts indicating potential reversals or continuations',
        difficulty: 'intermediate',
        resources: [
          { type: 'article', title: 'The Complete Guide to Chart Patterns' },
          { type: 'video', title: 'Top 5 Chart Patterns Every Trader Should Know' }
        ]
      },
      {
        name: 'Candlestick Patterns',
        description: 'Specific formations of Japanese candlesticks that signal potential market movements',
        difficulty: 'beginner',
        resources: [
          { type: 'article', title: 'Essential Candlestick Patterns for Traders' },
          { type: 'video', title: 'Mastering Candlestick Patterns' }
        ]
      },
      {
        name: 'Market Structure',
        description: 'Analysis of higher highs/lows and lower highs/lows to determine trend direction',
        difficulty: 'intermediate',
        resources: [
          { type: 'article', title: 'Understanding Market Structure' },
          { type: 'video', title: 'How to Trade with Market Structure' }
        ]
      }
    ]
  },
  {
    id: 'order_flow',
    name: 'Order Flow',
    description: 'Analysis of buying and selling pressure',
    icon: <Timeline />,
    color: '#ff9800',
    concepts: [
      {
        name: 'Order Blocks',
        description: 'Areas where significant buying or selling occurred, often from institutional players',
        difficulty: 'advanced',
        resources: [
          { type: 'article', title: 'Trading with Order Blocks' },
          { type: 'video', title: 'Order Block Strategy for Consistent Profits' }
        ]
      },
      {
        name: 'Liquidity',
        description: 'Areas where stop losses are clustered, often targeted by smart money',
        difficulty: 'advanced',
        resources: [
          { type: 'article', title: 'Understanding Liquidity in Trading' },
          { type: 'video', title: 'How to Identify and Trade Liquidity Zones' }
        ]
      },
      {
        name: 'Fair Value Gaps',
        description: 'Imbalances in price that occur during strong momentum moves',
        difficulty: 'advanced',
        resources: [
          { type: 'article', title: 'Trading Fair Value Gaps' },
          { type: 'video', title: 'How to Identify and Trade Fair Value Gaps' }
        ]
      }
    ]
  },
  {
    id: 'wyckoff',
    name: 'Wyckoff Method',
    description: 'Market cycle analysis based on the work of Richard Wyckoff',
    icon: <Psychology />,
    color: '#9c27b0',
    concepts: [
      {
        name: 'Accumulation',
        description: 'Phase where smart money accumulates positions before a markup phase',
        difficulty: 'expert',
        resources: [
          { type: 'article', title: 'Wyckoff Accumulation Pattern' },
          { type: 'video', title: 'How to Trade the Wyckoff Accumulation Pattern' }
        ]
      },
      {
        name: 'Distribution',
        description: 'Phase where smart money distributes positions before a markdown phase',
        difficulty: 'expert',
        resources: [
          { type: 'article', title: 'Wyckoff Distribution Pattern' },
          { type: 'video', title: 'How to Trade the Wyckoff Distribution Pattern' }
        ]
      },
      {
        name: 'Spring',
        description: 'A price move that briefly penetrates support in an accumulation phase',
        difficulty: 'expert',
        resources: [
          { type: 'article', title: 'Trading the Wyckoff Spring' },
          { type: 'video', title: 'How to Identify and Trade the Wyckoff Spring' }
        ]
      }
    ]
  },
  {
    id: 'elliott_wave',
    name: 'Elliott Wave Theory',
    description: 'Wave pattern analysis based on market psychology',
    icon: <TrendingUp />,
    color: '#4caf50',
    concepts: [
      {
        name: 'Impulse Waves',
        description: 'Five-wave structure in the direction of the trend',
        difficulty: 'expert',
        resources: [
          { type: 'article', title: 'Understanding Elliott Wave Impulse Patterns' },
          { type: 'video', title: 'How to Count Impulse Waves' }
        ]
      },
      {
        name: 'Corrective Waves',
        description: 'Three-wave structure against the trend',
        difficulty: 'expert',
        resources: [
          { type: 'article', title: 'Elliott Wave Corrective Patterns' },
          { type: 'video', title: 'Trading Corrective Waves' }
        ]
      },
      {
        name: 'Wave Counting',
        description: 'Identifying and labeling waves in a price chart',
        difficulty: 'expert',
        resources: [
          { type: 'article', title: 'Elliott Wave Counting Rules' },
          { type: 'video', title: 'Elliott Wave Counting for Beginners' }
        ]
      }
    ]
  },
  {
    id: 'volume_analysis',
    name: 'Volume Analysis',
    description: 'Study of volume and its relationship to price',
    icon: <BarChart />,
    color: '#e91e63',
    concepts: [
      {
        name: 'Volume Profile',
        description: 'Horizontal histogram showing trading activity at different price levels',
        difficulty: 'advanced',
        resources: [
          { type: 'article', title: 'Trading with Volume Profile' },
          { type: 'video', title: 'Volume Profile Trading Strategies' }
        ]
      },
      {
        name: 'VWAP',
        description: 'Volume Weighted Average Price, a benchmark used by institutional traders',
        difficulty: 'intermediate',
        resources: [
          { type: 'article', title: 'VWAP Trading Strategies' },
          { type: 'video', title: 'How to Use VWAP in Day Trading' }
        ]
      },
      {
        name: 'VSA',
        description: 'Volume Spread Analysis, examining the relationship between price spread and volume',
        difficulty: 'advanced',
        resources: [
          { type: 'article', title: 'Volume Spread Analysis Explained' },
          { type: 'video', title: 'VSA Trading Techniques' }
        ]
      }
    ]
  }
];

const KnowledgeBase = () => {
  const theme = useTheme();
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedCategory, setExpandedCategory] = useState(null);
  const [selectedConcept, setSelectedConcept] = useState(null);
  const [bookmarkedConcepts, setBookmarkedConcepts] = useState([]);

  const handleSearchChange = (event) => {
    setSearchTerm(event.target.value);
  };

  const handleCategoryExpand = (categoryId) => {
    setExpandedCategory(expandedCategory === categoryId ? null : categoryId);
  };

  const handleConceptSelect = (concept) => {
    setSelectedConcept(concept);
  };

  const handleBookmarkToggle = (concept) => {
    if (bookmarkedConcepts.some(c => c.name === concept.name)) {
      setBookmarkedConcepts(bookmarkedConcepts.filter(c => c.name !== concept.name));
    } else {
      setBookmarkedConcepts([...bookmarkedConcepts, concept]);
    }
  };

  const filteredCategories = KNOWLEDGE_CATEGORIES.filter(category => {
    if (!searchTerm) return true;
    
    const searchTermLower = searchTerm.toLowerCase();
    
    if (category.name.toLowerCase().includes(searchTermLower) || 
        category.description.toLowerCase().includes(searchTermLower)) {
      return true;
    }
    
    return category.concepts.some(concept => 
      concept.name.toLowerCase().includes(searchTermLower) || 
      concept.description.toLowerCase().includes(searchTermLower)
    );
  });

  const renderConceptDetail = () => {
    if (!selectedConcept) return null;
    
    return (
      <Card variant="outlined">
        <CardHeader 
          title={selectedConcept.name}
          subheader={`Difficulty: ${selectedConcept.difficulty}`}
          action={
            <IconButton 
              onClick={() => handleBookmarkToggle(selectedConcept)}
              color={bookmarkedConcepts.some(c => c.name === selectedConcept.name) ? "primary" : "default"}
            >
              {bookmarkedConcepts.some(c => c.name === selectedConcept.name) ? 
                <Bookmark /> : <BookmarkBorder />}
            </IconButton>
          }
        />
        <CardContent>
          <Typography variant="body1" paragraph>
            {selectedConcept.description}
          </Typography>
          
          <Typography variant="subtitle2" gutterBottom>
            Learning Resources
          </Typography>
          <List dense>
            {selectedConcept.resources.map((resource, index) => (
              <ListItem key={index}>
                <ListItemIcon>
                  {resource.type === 'article' ? <Info fontSize="small" /> : <School fontSize="small" />}
                </ListItemIcon>
                <ListItemText 
                  primary={resource.title}
                  secondary={resource.type}
                />
              </ListItem>
            ))}
          </List>
        </CardContent>
        <CardActions>
          <Button 
            size="small" 
            endIcon={<ArrowForward />}
          >
            Learn More
          </Button>
        </CardActions>
      </Card>
    );
  };

  return (
    <Paper elevation={0} variant="outlined">
      <Box p={2}>
        <Typography variant="h6" gutterBottom>Trading Knowledge Base</Typography>
        <Typography variant="body2" color="textSecondary" paragraph>
          Explore trading concepts and strategies to enhance your understanding
        </Typography>
        
        <TextField
          fullWidth
          placeholder="Search concepts, strategies, or patterns..."
          variant="outlined"
          value={searchTerm}
          onChange={handleSearchChange}
          margin="normal"
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search />
              </InputAdornment>
            ),
          }}
        />
        
        <Grid container spacing={3} mt={1}>
          <Grid item xs={12} md={selectedConcept ? 6 : 12}>
            {filteredCategories.map((category) => (
              <Accordion 
                key={category.id}
                expanded={expandedCategory === category.id}
                onChange={() => handleCategoryExpand(category.id)}
                sx={{ mb: 2 }}
              >
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Box display="flex" alignItems="center">
                    <Box 
                      sx={{ 
                        mr: 2, 
                        color: 'white', 
                        bgcolor: category.color,
                        borderRadius: '50%',
                        width: 40,
                        height: 40,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}
                    >
                      {category.icon}
                    </Box>
                    <Box>
                      <Typography variant="subtitle1">{category.name}</Typography>
                      <Typography variant="body2" color="textSecondary">
                        {category.description}
                      </Typography>
                    </Box>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <List dense>
                    {category.concepts.map((concept, index) => (
                      <React.Fragment key={index}>
                        <ListItem 
                          button
                          selected={selectedConcept && selectedConcept.name === concept.name}
                          onClick={() => handleConceptSelect(concept)}
                        >
                          <ListItemIcon>
                            <Lightbulb />
                          </ListItemIcon>
                          <ListItemText 
                            primary={concept.name}
                            secondary={concept.description}
                          />
                          <Chip 
                            label={concept.difficulty} 
                            size="small"
                            color={
                              concept.difficulty === 'beginner' ? 'success' :
                              concept.difficulty === 'intermediate' ? 'primary' :
                              concept.difficulty === 'advanced' ? 'warning' : 'error'
                            }
                          />
                        </ListItem>
                        {index < category.concepts.length - 1 && <Divider variant="inset" component="li" />}
                      </React.Fragment>
                    ))}
                  </List>
                </AccordionDetails>
              </Accordion>
            ))}
          </Grid>
          
          {selectedConcept && (
            <Grid item xs={12} md={6}>
              {renderConceptDetail()}
            </Grid>
          )}
        </Grid>
        
        {bookmarkedConcepts.length > 0 && (
          <Box mt={3}>
            <Typography variant="subtitle1" gutterBottom>
              Bookmarked Concepts
            </Typography>
            <Grid container spacing={2}>
              {bookmarkedConcepts.map((concept, index) => (
                <Grid item xs={12} sm={6} md={4} key={index}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="subtitle2">{concept.name}</Typography>
                      <Typography variant="body2" color="textSecondary" noWrap>
                        {concept.description}
                      </Typography>
                    </CardContent>
                    <CardActions>
                      <Button 
                        size="small" 
                        onClick={() => handleConceptSelect(concept)}
                      >
                        View
                      </Button>
                      <IconButton 
                        size="small" 
                        onClick={() => handleBookmarkToggle(concept)}
                        color="primary"
                      >
                        <Bookmark fontSize="small" />
                      </IconButton>
                    </CardActions>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Box>
        )}
      </Box>
    </Paper>
  );
};

export default KnowledgeBase;