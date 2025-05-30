import {
  Dashboard,
  TrendingUp,
  BarChart,
  Settings,
  Notifications,
  Psychology,
  Timeline,
  History,
  Assessment,
  Tune,
  Memory,
  ShowChart,
  Insights,
  BubbleChart,
  AutoGraph
} from '@mui/icons-material';

const navigationConfig = [
  {
    id: 'main',
    title: 'Main',
    type: 'group',
    children: [
      {
        id: 'dashboard',
        title: 'Dashboard',
        type: 'item',
        icon: Dashboard,
        url: '/',
        exact: true
      },
      {
        id: 'trading',
        title: 'Trading Terminal',
        type: 'item',
        icon: TrendingUp,
        url: '/trading'
      },
      {
        id: 'portfolio',
        title: 'Portfolio',
        type: 'item',
        icon: BarChart,
        url: '/portfolio'
      },
      {
        id: 'enhanced-intelligence',
        title: 'Enhanced Intelligence',
        type: 'item',
        icon: Psychology,
        url: '/enhanced-intelligence',
        badge: {
          title: 'NEW',
          color: 'secondary'
        }
      }
    ]
  },
  {
    id: 'analysis',
    title: 'Analysis',
    type: 'group',
    children: [
      {
        id: 'market-analysis',
        title: 'Market Analysis',
        type: 'item',
        icon: ShowChart,
        url: '/market-analysis'
      },
      {
        id: 'brain-performance',
        title: 'Brain Performance',
        type: 'item',
        icon: Insights,
        url: '/brain-performance'
      },
      {
        id: 'pattern-library',
        title: 'Pattern Library',
        type: 'item',
        icon: BubbleChart,
        url: '/pattern-library'
      },
      {
        id: 'ml-model-training',
        title: 'ML Model Training',
        type: 'item',
        icon: AutoGraph,
        url: '/ml-model-training'
      }
    ]
  },
  {
    id: 'tools',
    title: 'Tools',
    type: 'group',
    children: [
      {
        id: 'backtesting',
        title: 'Backtesting',
        type: 'item',
        icon: History,
        url: '/backtesting'
      },
      {
        id: 'strategy-builder',
        title: 'Strategy Builder',
        type: 'item',
        icon: Tune,
        url: '/strategy-builder'
      },
      {
        id: 'system-monitor',
        title: 'System Monitor',
        type: 'item',
        icon: Memory,
        url: '/system-monitor'
      }
    ]
  },
  {
    id: 'account',
    title: 'Account',
    type: 'group',
    children: [
      {
        id: 'notifications',
        title: 'Notifications',
        type: 'item',
        icon: Notifications,
        url: '/notifications'
      },
      {
        id: 'settings',
        title: 'Settings',
        type: 'item',
        icon: Settings,
        url: '/settings'
      }
    ]
  }
];

export default navigationConfig;
