/**!
* Written by Ascari Gutierrez Hermosillo <ascari.gtz@gmail.com>
* Enjoy!
*/

const np = require('./numpy')
const { LstmParam, LstmNetwork } = require('./lstm')

// Python's range() equivalent
const range = (a) => [...Array(a).keys()];

class ToyLossLayer {
  /*
  Computes square loss with first element of hidden layer array.
  */
  loss(pred, label) {
    return Math.pow((pred.get(0) - label), 2)
  }
  bottom_diff(pred, label) {
    let diff = np.zeros_like(pred)
    diff.set(0, 2 * (pred.get(0) - label))
    return diff
  }
}

function example_0() {
  // learns to repeat simple sequence from random inputs
  np.random.seed(0)

  // parameters for input data dimension and lstm cell count
  let mem_cell_ct = 100
  let x_dim = 50
  let lstm_param = new LstmParam(mem_cell_ct, x_dim)
  let lstm_net = new LstmNetwork(lstm_param)
  let y_list = [-0.5, 0.2, 0.1, -0.5]
  let input_val_arr = y_list.map(x => np.random.random(x_dim))

  for (cur_iter in range(10000)) {

    for (ind in range(y_list.length)) {
      lstm_net.x_list_add(input_val_arr[ind])
    }

    let loss = lstm_net.y_list_is(y_list, new ToyLossLayer())

    console.log(
      'iter',
      cur_iter,
      ':',
      'y_pred = [',
      range(y_list.length).map(ind => {
        const node = lstm_net.lstm_node_list[ind];
        return node.state.h.get(0).toFixed(5);
      }).join(','),
      ']',
      loss
    );

    lstm_param.apply_diff(0.1)
    lstm_net.x_list_clear()
  }
}

example_0();
