import { copyAndWatch } from '../lib/copyAndWatch.js';

const watch = !!process.argv[2];

copyAndWatch(
  [
    { src: 'public/**/*', srcPrefix: 'public', dst: 'out' },
    { src: 'sample/**/*', dst: 'out' },
  ],
  { watch }
);
