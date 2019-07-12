/**!
* Written by Ascari Gutierrez Hermosillo <ascari.gtz@gmail.com>
* Enjoy!
*/

// Adds numpy-like functionality that is missing in numjs

const nj = require('numjs')
const np = module.exports = nj

np.outer = function(a, b) {
  return np.array(a.tolist().map(Aa => b.tolist().map(Bb => Bb * Aa)))
}

np.hstack = function(a, b) {
  return np.concatenate(np.array(a).flatten(), np.array(b).flatten())
}

np.zeros_like = function(x) {
  return nj.zeros(x.shape)
}

const nj_random = nj.random

np.random = {
  rand: np.random,
  random: (shape) => shape ? nj_random(shape) : Math.random(),
  seed: (s) => np.random._seed = s // js doesn't seed its PRNG
}