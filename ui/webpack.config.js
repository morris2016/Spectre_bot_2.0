const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');
const CopyPlugin = require('copy-webpack-plugin');
const TerserPlugin = require('terser-webpack-plugin');
const DotenvPlugin = require('dotenv-webpack');
const { BundleAnalyzerPlugin } = require('webpack-bundle-analyzer');
const ReactRefreshWebpackPlugin = require('@pmmmwh/react-refresh-webpack-plugin');

module.exports = (env, argv) => {
  const isProduction = argv.mode === 'production';
  const isDevelopment = !isProduction;
  const shouldAnalyze = Boolean(env.analyze);

  return {
    mode: isProduction ? 'production' : 'development',
    entry: './src/index.js',
    output: {
      path: path.resolve(__dirname, 'dist'),
      filename: 'js/[name].[contenthash:8].js',
      chunkFilename: 'js/[name].[contenthash:8].chunk.js',
      assetModuleFilename: 'assets/[name].[contenthash:8][ext]',
      publicPath: '/',
      clean: true
    },
    devtool: isProduction ? 'source-map' : 'eval-cheap-module-source-map',
    resolve: {
      extensions: ['.ts', '.tsx', '.js', '.jsx', '.json'],
      alias: {
        '@': path.resolve(__dirname, 'src'),
        '@components': path.resolve(__dirname, 'src/components'),
        '@pages': path.resolve(__dirname, 'src/pages'),
        '@store': path.resolve(__dirname, 'src/store'),
        '@services': path.resolve(__dirname, 'src/services'),
        '@utils': path.resolve(__dirname, 'src/utils'),
        '@hooks': path.resolve(__dirname, 'src/hooks'),
        '@assets': path.resolve(__dirname, 'src/assets'),
        '@styles': path.resolve(__dirname, 'src/styles'),
        '@config': path.resolve(__dirname, 'src/config'),
        '@api': path.resolve(__dirname, 'src/api')
      },
      fallback: {
        "crypto": require.resolve('crypto-browserify'),
        "stream": require.resolve('stream-browserify'),
        "buffer": require.resolve('buffer/'),
        "path": require.resolve('path-browserify'),
        "os": require.resolve('os-browserify/browser')
      }
    },
    module: {
      rules: [
        {
          test: /\.(ts|tsx)$/,
          exclude: /node_modules/,
          use: [
            {
              loader: 'babel-loader',
              options: {
                presets: [
                  '@babel/preset-env',
                  ['@babel/preset-react', { runtime: 'automatic' }],
                  '@babel/preset-typescript'
                ],
                plugins: [
                  isDevelopment && require.resolve('react-refresh/babel'),
                  '@babel/plugin-transform-runtime',
                  '@babel/plugin-proposal-class-properties'
                ].filter(Boolean)
              }
            }
          ]
        },
        {
          test: /\.worker\.js$/,
          use: { loader: 'worker-loader' }
        },
        {
          test: /\.(css|scss|sass)$/,
          use: [
            isDevelopment ? 'style-loader' : MiniCssExtractPlugin.loader,
            {
              loader: 'css-loader',
              options: {
                importLoaders: 2,
                sourceMap: true
              }
            },
            {
              loader: 'postcss-loader',
              options: {
                postcssOptions: {
                  plugins: [
                    'postcss-preset-env',
                    'tailwindcss',
                    'autoprefixer'
                  ]
                },
                sourceMap: true
              }
            },
            {
              loader: 'sass-loader',
              options: {
                sourceMap: true
              }
            }
          ]
        },
        {
          test: /\.(png|svg|jpg|jpeg|gif)$/i,
          type: 'asset',
          parser: {
            dataUrlCondition: {
              maxSize: 8 * 1024 // 8kb
            }
          }
        },
        {
          test: /\.(woff|woff2|eot|ttf|otf)$/i,
          type: 'asset/resource'
        },
        {
          test: /\.(csv|tsv)$/i,
          use: ['csv-loader']
        },
        {
          test: /\.xml$/i,
          use: ['xml-loader']
        },
        {
          test: /\.mp3$/,
          use: {
            loader: 'file-loader',
            options: {
              name: '[name].[contenthash:8].[ext]',
              outputPath: 'assets/audio/'
            }
          }
        },
        {
          test: /\.glsl$/,
          use: 'raw-loader'
        }
      ]
    },
    optimization: {
      minimize: isProduction,
      minimizer: [
        new TerserPlugin({
          terserOptions: {
            parse: {
              ecma: 8
            },
            compress: {
              ecma: 5,
              warnings: false,
              comparisons: false,
              inline: 2
            },
            mangle: {
              safari10: true
            },
            output: {
              ecma: 5,
              comments: false,
              ascii_only: true
            }
          }
        })
      ],
      splitChunks: {
        chunks: 'all',
        cacheGroups: {
          vendor: {
            name: 'vendors',
            test: /[\\/]node_modules[\\/]/,
            chunks: 'all',
            priority: -10
          },
          charts: {
            name: 'charts',
            test: /[\\/]node_modules[\\/](chart\.js|lightweight-charts|d3|recharts)[\\/]/,
            chunks: 'all',
            priority: 20
          },
          tensorflow: {
            name: 'tensorflow',
            test: /[\\/]node_modules[\\/]@tensorflow[\\/]/,
            chunks: 'all',
            priority: 30
          }
        }
      },
      runtimeChunk: 'single'
    },
    plugins: [
      new HtmlWebpackPlugin({
        template: './public/index.html',
        favicon: './public/favicon.ico',
        inject: true,
        meta: {
          viewport: 'width=device-width, initial-scale=1, shrink-to-fit=no',
          'theme-color': '#000000'
        },
        minify: isProduction ? {
          removeComments: true,
          collapseWhitespace: true,
          removeRedundantAttributes: true,
          useShortDoctype: true,
          removeEmptyAttributes: true,
          removeStyleLinkTypeAttributes: true,
          keepClosingSlash: true,
          minifyJS: true,
          minifyCSS: true,
          minifyURLs: true
        } : false
      }),
      new DotenvPlugin({
        path: `./.env${isDevelopment ? '.development' : ''}`,
        safe: true,
        systemvars: true
      }),
      isDevelopment && new ReactRefreshWebpackPlugin(),
      isProduction && new MiniCssExtractPlugin({
        filename: 'css/[name].[contenthash:8].css',
        chunkFilename: 'css/[name].[contenthash:8].chunk.css'
      }),
      isProduction && new CopyPlugin({
        patterns: [
          { 
            from: 'public', 
            to: '.', 
            globOptions: {
              ignore: ['**/index.html', '**/favicon.ico']
            }
          }
        ]
      }),
      shouldAnalyze && new BundleAnalyzerPlugin()
    ].filter(Boolean),
    devServer: {
      static: path.join(__dirname, 'public'),
      historyApiFallback: true,
      compress: true,
      hot: true,
      port: 3000,
      client: {
        overlay: {
          errors: true,
          warnings: false
        }
      },
      proxy: {
        '/api': {
          target: 'http://localhost:5000',
          changeOrigin: true
        },
        '/ws': {
          target: 'ws://localhost:5000',
          ws: true
        }
      }
    },
    performance: {
      hints: isProduction ? 'warning' : false,
      maxEntrypointSize: 512000,
      maxAssetSize: 512000
    },
    stats: {
      children: false,
      modules: false
    },
    experiments: {
      topLevelAwait: true,
      asyncWebAssembly: true
    }
  };
};