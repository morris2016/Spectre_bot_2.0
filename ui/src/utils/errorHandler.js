// Basic global error logger
window.addEventListener('error', event => {
  console.error('Unhandled error:', event.error);
});
