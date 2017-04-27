package layers;

import java.util.Random;

import org.apache.commons.lang3.StringUtils;
import org.jblas.FloatMatrix;

import util.ActivationFunction;
import util.RandomGenerator;
import Mersenne.Sfmt;

/**
 * 全結合層
 * @author WinGAIA
 *
 */
public class Hiddenlayer {
	int input_N;
	int hidden_N;
	int output_N;

	FloatMatrix input_data;
	FloatMatrix output_data;

	public FloatMatrix weight;
//	public float[][] weight;
	public FloatMatrix bias;
//	public float[] bias;
	Sfmt mt;
	Random rng;
	ActivationFunction.FloatMatrixFunction<FloatMatrix> activation;
	ActivationFunction.FloatMatrixFunction<FloatMatrix> dactivation;
//	public FloatFunction<Float> activation;
//	public FloatFunction<Float> dactivation;

	//モーメンタム、momentum*前のgrad_weightを現時点の重みの更新量に足す
	float momentum;
	//重みの減衰、l_rate*w_decay*weightを現時点の重みの更新量から引く
	float w_decay;

//	public DoubleFunction<Double> double_activation;
//	public DoubleFunction<Double> double_dactivation;

	@FunctionalInterface
	public interface FloatFunction<R> {

		/**
		 * Applies this function to the given argument.
		 *
		 * @param value the function argument
		 * @return the function result
		 */
		R apply(float value);
	}
	/**
	 * 隠れ層のコンストラクタ
	 * @param input インプットの数
	 * @param output アウトプットの数
	 * @param W 重み
	 * @param b バイアス
	 * @param m メルセンヌツイスターのインスタンス
	 * @param actfunc 活性化関数
	 */
	public Hiddenlayer(int input, int output, FloatMatrix W, FloatMatrix b, int minibatch, Sfmt m, String actfunc) {
//		public Hiddenlayer(int input, int output, float[][] W, float[] b, Sfmt m, String actfunc) {
		// TODO 自動生成されたコンストラクター・スタブ
		if(m == null){
			int[] init_key = {(int) System.currentTimeMillis(), (int) Runtime.getRuntime().freeMemory()};
			m = new Sfmt(init_key);
		}

		//重みの初期化
		if(W == null){
			W = new FloatMatrix(output,input);
			float min = 1.f/ input;

			for(int i=0; i<output; i++){
				for(int j=0; j<input; j++){
					W.put(i, j, RandomGenerator.uniform(-min, min, m));
				}
			}
		}

		//バイアスの初期化
		if(b == null){
			b = new FloatMatrix(output);
		}

		input_N= input;
		output_N=output;
		weight=W;
		bias=b;
		mt=m;

		input_data = new FloatMatrix(minibatch, input);
		output_data = new FloatMatrix(minibatch, output);

		System.out.println("Creat Hidden Layer... input:"+input+" output:"+output);
		if(actfunc.equals("sigmoid")){
			activation = (FloatMatrix x)->ActivationFunction.logistic_sigmoid(x);
			dactivation = (FloatMatrix x)->ActivationFunction.dsigmoid(x);
		}else if(actfunc.equals("tanh")){
			activation = (FloatMatrix x)->ActivationFunction.tanh(x);
			dactivation = (FloatMatrix x)->ActivationFunction.dtanh(x);
		}else if(actfunc.equals("ReLU")){
			activation = (FloatMatrix x)->ActivationFunction.ReLU(x);
			dactivation = (FloatMatrix x)->ActivationFunction.dReLU(x);
		}else if(StringUtils.isEmpty(actfunc)){
			throw new IllegalArgumentException("specify activation function");
		}else{
			throw new IllegalArgumentException("activation function not supported");
		}
	}

	/*public Hiddenlayer(int input, int output, float[][] W, float[] b, Random rn, String actfunc) {
		if (rn == null) rn = new Random(1234);  // seed random

		if (W == null) {

			W = new float[output][input];
			float w_ = 1.f / input;

			for(int j = 0; j < output; j++) {
				for(int i = 0; i < input; i++) {
					W[j][i] = RandomGenerator.uniform(-w_, w_, rn);  // initialize W with uniform distribution
				}
			}

		}

		if (b == null) b = new float[output];

		this.input_N = input;
		this.output_N = output;
		this.weight = W;
		this.bias = b;
		this.rng = rn;

		if (actfunc == "sigmoid" || activation == null) {

			this.double_activation = (double x) -> ActivationFunction.logistic_sigmoid(x);
			this.double_dactivation = (double x) -> ActivationFunction.dsigmoid(x);

		} else if (actfunc == "tanh") {

			this.double_activation = (double x) -> ActivationFunction.tanh(x);
			this.double_dactivation = (double x) -> ActivationFunction.dtanh(x);

		} else if (actfunc == "ReLU") {

			this.double_activation = (double x) -> ActivationFunction.ReLU(x);
			this.double_dactivation = (double x) -> ActivationFunction.dReLU(x);

		} else {
			throw new IllegalArgumentException("activation function not supported");
		}
	}*/

	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ

	}

	/**
	 * 順伝播
	 * @param x 入力データ
	 * @return 出力
	 */
	public FloatMatrix forward(FloatMatrix x) {
//		public float[] forward(float[] x) {
		// TODO 自動生成されたメソッド・スタブ
//		System.out.println(x.rows +":"+x.columns+":");
//		System.out.println(x);
		return output(x);
	}

	public FloatMatrix output(FloatMatrix x) {
		// TODO 自動生成されたメソッド・スタブ
//		FloatMatrix y = new FloatMatrix(x.rows,output_N);
//		System.out.println("x:"+ x.rows+"**"+x.columns+"##"+output_N);
//		System.out.println("w:"+ weight.rows+"**"+weight.columns+"##"+output_N);

		return activation.apply(x.mmul(weight.transpose()).addRowVector(bias));
		/*for(int i=0; i<output_N; i++){
			float act = 0.f;
			//y[i] = 0.f;
			for(int j=0; j<input_N; j++){
				//y[i] += weight[i][j] * x[j];
				act += weight[i][j] * x[j];
			}
			//y[i] += bias[i];
			act += bias[i];
			y[i] = activation.apply(act);
		}*/

//		return y;
	}

	/**
	 * 全結合層の順伝播
	 * @param x 入力データ
	 * @param n ミニバッチ番号、全部投入はミニバッチ数を、負ならば入力値に代入しない
	 * @return アウトプット結果
	 */
	public FloatMatrix forward(FloatMatrix x, int n) {

		if(n == input_data.rows){
			input_data = x.dup();
			output_data = output(x);
		}else if(n >= 0){
			input_data.putRow(n, x.dup());
			output_data.putRow(n, output(x));
			return output_data.getRow(n);
		}else{
			return output(x);
		}

		return output_data;
	}

	public FloatMatrix output(FloatMatrix x,int n) {
		// TODO 自動生成されたメソッド・スタブ
		return activation.apply(x.mmul(weight.transpose()).addRowVector(bias));
	}

	/**
	 * 順伝播
	 * @param x 入力データ
	 * @return 出力

	public double[] forward(double[] x) {
		// TODO 自動生成されたメソッド・スタブ
		return output(x);
	}

	public double[] output(double[] x) {
		// TODO 自動生成されたメソッド・スタブ
		double y[] = new double[output_N];

		for(int i=0; i<output_N; i++){
			float act = 0.f;
			//y[i] = 0.f;
			for(int j=0; j<input_N; j++){
				//y[i] += weight[i][j] * x[j];
				act += weight[i][j] * x[j];
			}
			//y[i] += bias[i];
			act += bias[i];
			y[i] = double_activation.apply(act);
		}

		return y;
	}*/

	public FloatMatrix outputBinomial(FloatMatrix x, Sfmt mt) {
//		public int[] outputBinomial(int[] x, Sfmt mt) {

//		FloatMatrix y = new FloatMatrix(output_N);
//		int[] y = new int[output_N];

		FloatMatrix xCast = x.dup();
//		float[] xCast = new float(x.length);
//		for (int i = 0; i < xCast.length; i++) {
//			xCast[i] = (float) x[i];
//		}

		FloatMatrix out = output(xCast);
/*		float[] out = output(xCast);

		for (int j = 0; j < output_N; j++) {
			y[j] = RandomGenerator.binomial(1, out[j], mt);
		}
*/
		FloatMatrix y = RandomGenerator.binomial(1, out, mt);
		return y;
	}

	/**
	 * 逆伝播メソッド
	 * @param x その層への入力
	 * @param z その層の出力
	 * @param dy 前の層（出力に近い層）の逆伝播の値
	 * @param weight_prev 前の層への重み
	 * @param minibatchSize ミニバッチサイズ
	 * @param l_rate 学習率
	 * @return 逆伝播の結果
	 */
	public FloatMatrix backward(FloatMatrix x, FloatMatrix z, FloatMatrix dy, FloatMatrix weight_prev, int minibatchSize, float l_rate) {
//		public float[][] backward(float[][] x, float[][] z, float[][] dy, float[][] weight_prev, int minibatchSize, float l_rate) {
		// TODO 自動生成されたメソッド・スタブ
		//逆伝播する誤差
		FloatMatrix dz = new FloatMatrix(minibatchSize, output_N);
//		float[][] dz = new float[minibatchSize][output_N];

		FloatMatrix grad_w = new FloatMatrix(output_N, input_N);
		FloatMatrix grad_b = new FloatMatrix(output_N);
//		float[][] grad_w = new float[output_N][input_N];
//		float[] grad_b = new float[output_N];

//		System.out.println("z:"+ z.rows+"**"+z.columns+"##"+output_N);
//		System.out.println("dz"+ dz.rows+"**"+dz.columns+"##"+output_N);

		//誤差を計算
		dz = dy.mmul(weight_prev).dup();
		//出力を微分
		dz.muli(dactivation.apply(z));

		//重み誤差とバイアス誤差を計算
		grad_b = dz.columnSums().transpose().dup();
		grad_w = dz.transpose().mmul(x).dup();

//		System.out.println("grad"+ grad_w.rows+"**"+grad_w.columns);
//		System.out.println("weight"+ weight.rows+"**"+weight.columns+"##"+output_N);
//		System.out.println(z);
//		System.out.println(x);

		//重み更新
		weight.subi(grad_w.mul(l_rate).div((float)minibatchSize));
		bias.subi(grad_b.mul(l_rate).div((float)minibatchSize));

		//SGD
		/*for(int n=0; n<minibatchSize; n++){
			for(int i=0; i<output_N; i++){
				//前の層のアウトプット数
				for(int p=0; p<dy[0].length; p++){
					//前の層からの逆伝播
					//System.out.println(dy[0].length+"DDDD"+p);
					//System.out.println(":::"+weight_prev[p].length);
					//System.out.println(weight_prev.length);
					//System.out.println(dy[n][p]);
					dz[n][i] += weight_prev[p][i] * dy[n][p];
				}
				//出力を微分
				dz[n][i] *= dactivation.apply(z[n][i]);

				for(int j=0; j<input_N; j++){
					grad_w[i][j] += dz[n][i] * x[n][j];
				}
				grad_b[i] += dz[n][i];
			}

		}
		//重みとバイアスを更新
		for(int i=0; i<output_N; i++){
			for(int j=0; j<input_N; j++){
				//weight[i][j] = weight[i][j] - (l_rate * grad_w[i][j] / minibatchSize);
				weight[i][j] -= l_rate * grad_w[i][j] / minibatchSize;
			}
			//bias[i] = bias[i] - (l_rate * grad_b[i] / minibatchSize);
			bias[i] -= l_rate * grad_b[i] / minibatchSize;
		}*/

		return dz;
	}




	public FloatMatrix backward(FloatMatrix dy, FloatMatrix weight_prev, int minibatchSize, float l_rate) {
		FloatMatrix dz = new FloatMatrix(minibatchSize, output_N);

		FloatMatrix grad_w = new FloatMatrix(output_N, input_N);
		FloatMatrix grad_b = new FloatMatrix(output_N);

		//誤差を計算
		dz = dy.mmul(weight_prev).dup();
		//出力を微分
		dz.muli(dactivation.apply(output_data));

		//重み誤差とバイアス誤差を計算
		grad_b = dz.columnSums().transpose().dup();
		grad_w = dz.transpose().mmul(input_data).dup();
		//重み更新SGD
		weight.subi(grad_w.mul(l_rate).div((float)minibatchSize));
		bias.subi(grad_b.mul(l_rate).div((float)minibatchSize));

		return dz;
	}

	/**
	 * 逆伝播メソッド
	 * @param x その層への入力
	 * @param z その層の出力
	 * @param dy 前の層（出力に近い層）の逆伝播の値
	 * @param weight_prev 前の層への重み
	 * @param minibatchSize ミニバッチサイズ
	 * @param l_rate 学習率
	 * @return 逆伝播の結果

	public double[][] backward(double[][] x, double[][] z, double[][] dy,
			float[][] weight_prev, int minibatchSize, double l_rate) {
		// TODO 自動生成されたメソッド・スタブ
		//逆伝播する誤差
		double[][] dz = new double[minibatchSize][output_N];

		double[][] grad_w = new double[output_N][input_N];
		double[] grad_b = new double[output_N];

		//SGD
		for(int n=0; n<minibatchSize; n++){
			for(int i=0; i<output_N; i++){
				//前の層のアウトプット数
				for(int p=0; p<dy[0].length; p++){
					//前の層からの逆伝播
					dz[n][i] += weight_prev[p][i] * dy[n][p];
				}
				//出力を微分
				dz[n][i] *= double_dactivation.apply(z[n][i]);

				for(int j=0; j<input_N; j++){
					grad_w[i][j] += dz[n][i] * x[n][j];
				}
				grad_b[i] += dz[n][i];
			}
		}
		//重みとバイアスを更新
		for(int i=0; i<output_N; i++){
			for(int j=0; j<input_N; j++){
				weight[i][j] = (float) (weight[i][j] - (l_rate * grad_w[i][j] / minibatchSize));
			}
			bias[i] = (float) (bias[i] - (l_rate * grad_b[i] / minibatchSize));
		}


		return dz;
	}*/
}
