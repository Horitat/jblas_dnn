package layers;

import org.jblas.FloatMatrix;

import util.ActivationFunction;

public class Globalaveragepooling {
	FloatMatrix[][] input_data;
	FloatMatrix output_data;

	ActivationFunction.FloatMatrixFunction<FloatMatrix> activation;
	ActivationFunction.FloatMatrixFunction<FloatMatrix> dactivation;
	int minibatch;

	public Globalaveragepooling(int channel,int minibatch){
		input_data = new FloatMatrix[minibatch][channel];
		output_data = new FloatMatrix(minibatch,channel);
		this.minibatch = minibatch;
	}

	/**
	 * 順伝播、出力層でない場合
	 * @param z
	 * @return
	 */
	public FloatMatrix forward(FloatMatrix[][] z){
		input_data = z.clone();

		for(int m=0; m< z.length; m++){
			for(int c=0; c<z[m].length; c++){
				output_data.put(m, c, z[m][c].mean());
			}
		}
		return output_data.dup();
	}

	/**
	 * 出力層の場合
	 * @param z
	 * @param label
	 * @return 誤差
	 */
	public FloatMatrix[][] output(FloatMatrix[][] z,FloatMatrix label){
		return backward(ActivationFunction.softmax(forward(z),output_data.columns).sub(label));
	}


	public FloatMatrix[][] backward(FloatMatrix dy){
		FloatMatrix[][] back = new FloatMatrix[input_data.length][input_data[0].length];
		for(int m=0; m< back.length; m++){
			for(int c=0; c<back[m].length; c++){
				back[m][c] =
						new FloatMatrix(input_data[m][c].rows, input_data[m][c].columns).add(dy.get(m,c)/input_data[m][c].length).mul(output_data.get(m,c));
//				input_data[m][c].eq(output_data.get(m,c)).mul(dy.get(m,c)/input_data[m][c].length);
			}
		}
		return back;
	}

}
