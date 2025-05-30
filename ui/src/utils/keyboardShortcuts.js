export function setupKeyboardShortcuts(actions) {
  function handler(e) {
    if (e.key === '?') {
      actions.showShortcuts?.();
    }
  }
  window.addEventListener('keydown', handler);
  return () => window.removeEventListener('keydown', handler);
}
