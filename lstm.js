/**!
* Written by Ascari Gutierrez Hermosillo <ascari.gtz@gmail.com>
* Enjoy!
*/

const np = require('./numpy');

function sigmoid(x) {
  return np.ones(x.shape).divide(
    np.ones(x.shape).add(np.exp(x.multiply(-1)))
  );
}

function sigmoid_derivative(values) {
  return values.multiply(np.ones(values.shape).subtract(values))
}

function tanh_derivative(values) {
  return np.ones(values.shape).subtract(values.pow(2))
}

// createst uniform random array w/ values in [a,b) and shape args
function rand_arr(a, b, args=[]) {
  np.random.seed(0)
  return np.random.rand(args).multiply(b - a).add(a);
}

class LstmParam {
  constructor(mem_cell_ct, x_dim) {
    this.mem_cell_ct = mem_cell_ct
    this.x_dim = x_dim
    let concat_len = x_dim + mem_cell_ct
    // weight matrices
    this.wg = rand_arr(-0.1, 0.1, [mem_cell_ct, concat_len])
    this.wi = rand_arr(-0.1, 0.1, [mem_cell_ct, concat_len])
    this.wf = rand_arr(-0.1, 0.1, [mem_cell_ct, concat_len])
    this.wo = rand_arr(-0.1, 0.1, [mem_cell_ct, concat_len])
    // bias terms
    this.bg = rand_arr(-0.1, 0.1, [mem_cell_ct]) 
    this.bi = rand_arr(-0.1, 0.1, [mem_cell_ct]) 
    this.bf = rand_arr(-0.1, 0.1, [mem_cell_ct]) 
    this.bo = rand_arr(-0.1, 0.1, [mem_cell_ct]) 
    // diffs (derivative of loss function w.r.t. all parameters)
    this.wg_diff = np.zeros([mem_cell_ct, concat_len]) 
    this.wi_diff = np.zeros([mem_cell_ct, concat_len]) 
    this.wf_diff = np.zeros([mem_cell_ct, concat_len]) 
    this.wo_diff = np.zeros([mem_cell_ct, concat_len]) 
    this.bg_diff = np.zeros(mem_cell_ct) 
    this.bi_diff = np.zeros(mem_cell_ct) 
    this.bf_diff = np.zeros(mem_cell_ct) 
    this.bo_diff = np.zeros(mem_cell_ct) 
  }
  apply_diff(lr = 1) {
    this.wg =  this.wg.subtract(this.wg_diff.multiply(lr))
    this.wi =  this.wi.subtract(this.wi_diff.multiply(lr))
    this.wf =  this.wf.subtract(this.wf_diff.multiply(lr))
    this.wo =  this.wo.subtract(this.wo_diff.multiply(lr))
    this.bg =  this.bg.subtract(this.bg_diff.multiply(lr))
    this.bi =  this.bi.subtract(this.bi_diff.multiply(lr))
    this.bf =  this.bf.subtract(this.bf_diff.multiply(lr))
    this.bo =  this.bo.subtract(this.bo_diff.multiply(lr))
    // reset diffs to zero
    this.wg_diff = np.zeros_like(this.wg)
    this.wi_diff = np.zeros_like(this.wi)
    this.wf_diff = np.zeros_like(this.wf)
    this.wo_diff = np.zeros_like(this.wo)
    this.bg_diff = np.zeros_like(this.bg)
    this.bi_diff = np.zeros_like(this.bi)
    this.bf_diff = np.zeros_like(this.bf)
    this.bo_diff = np.zeros_like(this.bo)
  }
}

class LstmState {
  constructor(mem_cell_ct, x_dim) {
    this.g = np.zeros(mem_cell_ct)
    this.i = np.zeros(mem_cell_ct)
    this.f = np.zeros(mem_cell_ct)
    this.o = np.zeros(mem_cell_ct)
    this.s = np.zeros(mem_cell_ct)
    this.h = np.zeros(mem_cell_ct)
    this.bottom_diff_h = np.zeros_like(this.h)
    this.bottom_diff_s = np.zeros_like(this.s)    
  }
}

class LstmNode {
  constructor(lstm_param, lstm_state) {
    // store reference to parameters and to activations
    this.state = lstm_state
    this.param = lstm_param
    // non-recurrent input concatenated with recurrent input
    this.xc = null
  }
  bottom_data_is(x, s_prev = null, h_prev = null) {
    // if this is the first lstm node in the network
    if (s_prev == null) s_prev = np.zeros_like(this.state.s)
    if (h_prev == null) h_prev = np.zeros_like(this.state.h)
    // save data for use in backprop
    this.s_prev = s_prev
    this.h_prev = h_prev

    // concatenate x(t) and h(t-1)
    let xc = np.hstack(x,  h_prev)
    this.state.g = np.tanh(np.dot(this.param.wg, xc).add(this.param.bg))
    this.state.i = sigmoid(np.dot(this.param.wi, xc).add(this.param.bi))
    this.state.f = sigmoid(np.dot(this.param.wf, xc).add(this.param.bf))
    this.state.o = sigmoid(np.dot(this.param.wo, xc).add(this.param.bo))
    this.state.s = (this.state.g.multiply(this.state.i)).add(s_prev.multiply(this.state.f))
    this.state.h = this.state.s.multiply(this.state.o)

    this.xc = xc
  }
  top_diff_is(top_diff_h, top_diff_s) {
    // notice that top_diff_s is carried along the constant error carousel
    let ds = this.state.o.multiply(top_diff_h).add(top_diff_s)
    let _do = this.state.s.multiply(top_diff_h)
    let di = this.state.g.multiply(ds)
    let dg = this.state.i.multiply(ds)
    let df = this.s_prev.multiply(ds)

    // diffs w.r.t. vector inside sigma / tanh function
    let di_input = sigmoid_derivative(this.state.i).multiply(di)
    let df_input = sigmoid_derivative(this.state.f).multiply(df)
    let do_input = sigmoid_derivative(this.state.o).multiply(_do)
    let dg_input = tanh_derivative(this.state.g).multiply(dg)

    // diffs w.r.t. inputs
    this.param.wi_diff = this.param.wi_diff.add(np.outer(di_input, this.xc))
    this.param.wf_diff = this.param.wf_diff.add(np.outer(df_input, this.xc))
    this.param.wo_diff = this.param.wo_diff.add(np.outer(do_input, this.xc))
    this.param.wg_diff = this.param.wg_diff.add(np.outer(dg_input, this.xc))
    this.param.bi_diff = this.param.bi_diff.add(di_input)
    this.param.bf_diff = this.param.bf_diff.add(df_input)
    this.param.bo_diff = this.param.bo_diff.add(do_input)
    this.param.bg_diff = this.param.bg_diff.add(dg_input)

    // compute bottom diff
    let dxc = np.zeros_like(this.xc)
    dxc = dxc.add(np.dot(this.param.wi.T, di_input))
    dxc = dxc.add(np.dot(this.param.wf.T, df_input))
    dxc = dxc.add(np.dot(this.param.wo.T, do_input))
    dxc = dxc.add(np.dot(this.param.wg.T, dg_input))

    // save bottom diffs
    this.state.bottom_diff_s = ds.multiply(this.state.f)
    this.state.bottom_diff_h = dxc.slice(this.param.x_dim)
  }
}

class LstmNetwork {
  constructor(lstm_param) {
    this.lstm_param = lstm_param
    this.lstm_node_list = []
    // input sequence
    this.x_list = []
  }
  y_list_is(y_list, loss_layer) {
    /*
    Updates diffs by setting target sequence 
    with corresponding loss layer. 
    Will *NOT* update parameters.  To update parameters,
    call this.lstm_param.apply_diff()
    */
    require('assert')(y_list.length == this.x_list.length)

    let idx = this.x_list.length - 1
    // first node only gets diffs from label ...
    let loss = loss_layer.loss(this.lstm_node_list[idx].state.h, y_list[idx])
    let diff_h = loss_layer.bottom_diff(this.lstm_node_list[idx].state.h, y_list[idx])
    // here s is not affecting loss due to h(t+1), hence we set equal to zero
    let diff_s = np.zeros(this.lstm_param.mem_cell_ct)
    this.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
    idx -= 1

    // ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
    // we also propagate error along constant error carousel using diff_s
    while (idx >= 0) {
      loss += loss_layer.loss(this.lstm_node_list[idx].state.h, y_list[idx])
      diff_h = loss_layer.bottom_diff(this.lstm_node_list[idx].state.h, y_list[idx])
      diff_h = diff_h.add(this.lstm_node_list[idx + 1].state.bottom_diff_h)
      diff_s = this.lstm_node_list[idx + 1].state.bottom_diff_s
      this.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
      idx -= 1 
    }

    return loss
  }
  x_list_clear() {
    this.x_list = []
  }
  x_list_add(x) {
    this.x_list.push(x)
    if (this.x_list.length > this.lstm_node_list.length) {
      // need to add new lstm node, create new state mem
      let lstm_state = new LstmState(this.lstm_param.mem_cell_ct, this.lstm_param.x_dim)
      this.lstm_node_list.push(new LstmNode(this.lstm_param, lstm_state))
    }

    // get index of most recent x input
    let idx = this.x_list.length - 1
    if (idx == 0) {
      // no recurrent inputs yet
      this.lstm_node_list[idx].bottom_data_is(x)
    } else {
      let s_prev = this.lstm_node_list[idx - 1].state.s
      let h_prev = this.lstm_node_list[idx - 1].state.h
      this.lstm_node_list[idx].bottom_data_is(x, s_prev, h_prev)
    }
  }
}

module.exports = {
  LstmParam,
  LstmState,
  LstmNode,
  LstmNetwork,
};