import { nodeResolve } from '@rollup/plugin-node-resolve';
import typescript from '@rollup/plugin-typescript';
import commonjs from '@rollup/plugin-commonjs';
import { readDirSyncRecursive } from './build/lib/readdir.js';

const nodeVersion = parseInt(process.version.substring(1));
if (isNaN(nodeVersion) || nodeVersion < 20) {
  console.error('need node >= v20');
  process.exit(1);
}

const outPath = 'out';

function wgslPlugin() {
  return {
    name: 'wgsl-plugin',
    transform(code, id) {
      if (id.endsWith('.wgsl')) {
        return {
          code: `export default \`${code}\`;`,
          map: { mappings: '' },
        };
      }
    },
  };
}

const samplePlugins = [
  wgslPlugin(),
  nodeResolve(),
  commonjs(),
  typescript({ tsconfig: './sample/tsconfig.json' }),
];

const sampleFiles = readDirSyncRecursive('sample');

const samples = sampleFiles
  .filter((n) => n.endsWith('/main.ts'))
  .map((filename) => {
    return {
      input: filename,
      output: [
        {
          file: `${outPath}/${filename.replace(/\.ts$/, '.js')}`,
          format: 'esm',
          sourcemap: true,
        },
      ],
      plugins: samplePlugins,
    };
  });

export default [...samples];
