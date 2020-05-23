function inv(x) {
    if(typeof x === 'object') {
        return x.map(function(y) {
            return inv(y);
        });
    } else {
        return (1 / x).toFixed(8);
    }
}
function mult(x, y) {
    return x.map(function(v, k) {
        if(typeof x === 'object') {
            return v.map(function(v2, k2) {
                return (v2 * y[k][k2]).toFixed(8);
            });
        } else {
            return (v * y[k]).toFixed(8);
        }
    });
}
function NeuralNetwork() {
    this.layers           = [];
    this.weights          = [];
    this.training_inputs  = [];
    this.training_outputs = [[]];

    this.addLayer = function(inx , out) {
        var w = [];
        for(var i = 0; i < inx; i++) {
            var z = [];
            for(var j = 0; j < out; j++) {
                z.push((Math.random() * (0.00000000 - 1.00000000) + 1.00000000).toFixed(8));
            }
            w.push(z);
        }
        w = math.subtract(math.multiply(w, 2), 1);
        this.weights.push(w);
        this.layers.push([inx, out]);
        return;
    };
    this.learn = function(x, y) {
        if(x.length != this.layers[0][0]) {
            throw new Error('La cantidad de entrada no concuerda.');
        }
        this.training_inputs.push(x);
        this.training_outputs[0].push(y);
    };
    this.__sigmoid = function(x) {
        return inv(math.add(math.exp(math.multiply(x, -1)), 1));
    };
    this.__sigmoid_derivative = function(x) {
        return mult(x, math.subtract(1, x));
    };
    this.train = function(number_of_training_iterations) {
        outs = math.transpose(this.training_outputs);
        for(var i = 0; i < number_of_training_iterations; i++) {
            var output = this._think(this.training_inputs);
            var error  = {};
            var delta  = {};
            var adjus  = {};
            for(var j = this.layers.length - 1; j >= 0; j--) {
                if(j == this.layers.length - 1) {
                    error[j] = math.subtract(outs, output[this.layers.length - 1]);
                } else {
                    error[j] = math.multiply(delta[j + 1], math.transpose(this.weights[j + 1]));
                }
                delta[j] = mult(error[j], this.__sigmoid_derivative(output[j]));
                if(j == 0) {
                    adjus[j] = math.multiply(math.transpose(this.training_inputs), delta[j]);
                } else {
                    adjus[j] = math.multiply(math.transpose(output[this.layers.length - 2]), delta[j]);
                }
            }
            for(var j = 0; j < this.layers.length; j++) {
                this.weights[j] = math.add(this.weights[j], adjus[j]);
            }
        }
    };
    this._think = function(x) {
        var out = [];
        var ant = null;
        ant = this.__sigmoid(math.multiply(x, this.weights[0]));
        out.push(ant);
        for(var i = 1; i < this.layers.length; i++) {
            ant = this.__sigmoid(math.multiply(ant, this.weights[i]));
            out.push(ant);
        }
        return out;
    };
    this.think = function(x) {
        return this._think([x]).pop();
    };
}
