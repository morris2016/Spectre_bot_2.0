export function detectHardwareCapabilities() {
  return {
    hasWebGL: !!window.WebGLRenderingContext,
  };
}
