import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Drawer, List, ListItemButton, ListItemIcon, ListItemText, Collapse, Toolbar, useTheme, useMediaQuery } from '@mui/material';
import { ExpandLess, ExpandMore } from '@mui/icons-material';
import navigationConfig from '../config/navigationConfig';

const drawerWidth = 260;

const Sidebar = ({ open, onClose }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const theme = useTheme();
  const isMdUp = useMediaQuery(theme.breakpoints.up('md'));
  const [openMenus, setOpenMenus] = useState({});

  const handleToggle = (label) => {
    setOpenMenus((prev) => ({ ...prev, [label]: !prev[label] }));
  };

  const handleNavigate = (path) => {
    navigate(path);
    if (!isMdUp && onClose) {
      onClose();
    }
  };

  const renderItem = (item) => {
    if (item.subItems) {
      const expanded = openMenus[item.label] || false;
      const Icon = item.icon;
      return (
        <React.Fragment key={item.label}>
          <ListItemButton onClick={() => handleToggle(item.label)}>
            {Icon && (
              <ListItemIcon>
                <Icon />
              </ListItemIcon>
            )}
            <ListItemText primary={item.label} />
            {expanded ? <ExpandLess /> : <ExpandMore />}
          </ListItemButton>
          <Collapse in={expanded} timeout="auto" unmountOnExit>
            <List component="div" disablePadding>
              {item.subItems.map((sub) => {
                const SubIcon = sub.icon;
                return (
                  <ListItemButton
                    key={sub.label}
                    sx={{ pl: 4 }}
                    selected={location.pathname === sub.path}
                    onClick={() => handleNavigate(sub.path)}
                  >
                    {SubIcon && (
                      <ListItemIcon>
                        <SubIcon />
                      </ListItemIcon>
                    )}
                    <ListItemText primary={sub.label} />
                  </ListItemButton>
                );
              })}
            </List>
          </Collapse>
        </React.Fragment>
      );
    }
    const Icon = item.icon;
    return (
      <ListItemButton
        key={item.label}
        selected={location.pathname === item.path}
        onClick={() => handleNavigate(item.path)}
      >
        {Icon && (
          <ListItemIcon>
            <Icon />
          </ListItemIcon>
        )}
        <ListItemText primary={item.label} />
      </ListItemButton>
    );
  };

  return (
    <Drawer
      variant={isMdUp ? 'persistent' : 'temporary'}
      open={open}
      onClose={onClose}
      sx={{ '& .MuiDrawer-paper': { width: drawerWidth } }}
    >
      <Toolbar />
      <List>{navigationConfig.map(renderItem)}</List>
    </Drawer>
  );
};

export default Sidebar;
