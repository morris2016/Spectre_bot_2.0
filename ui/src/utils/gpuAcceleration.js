export function initializeGPUAcceleration() {
  if (navigator.gpu) {
    console.log('GPU acceleration available');
  } else {
    console.log('GPU acceleration not supported');
  }
}
