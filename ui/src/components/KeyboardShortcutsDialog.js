import React from 'react';

const KeyboardShortcutsDialog = ({ open, onClose }) => {
  if (!open) return null;
  return (
    <div onClick={onClose}>
      Keyboard Shortcuts
    </div>
  );
};

export default KeyboardShortcutsDialog;
