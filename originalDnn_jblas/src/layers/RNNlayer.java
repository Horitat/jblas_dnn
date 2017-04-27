package layers;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.StringUtils;
import org.jblas.FloatMatrix;

import util.ActivationFunction;
import util.Lossfunction;
import util.RandomGenerator;
import Mersenne.Sfmt;

public class RNNlayer {
	int input_N;
	int hidden_N;
	int output_N;
	int selfloop_output;

	public FloatMatrix weight;
	public FloatMatrix reccurent_weight;
	public FloatMatrix bias;
	public FloatMatrix recurrent_bias;
//	public float[][] weight;
//	public float[][] reccurent_weight;
//	public float[] bias;
//	public float[] recurrent_bias;
	Sfmt mt;
	public ActivationFunction.FloatMatrixFunction<FloatMatrix> activation;
	public ActivationFunction.FloatMatrixFunction<FloatMatrix> dactivation;
	public Lossfunction.FloatFunction<Float,Float> dlossfunc;

	//次に入力されるループ用
	private FloatMatrix loop_input;
//	private float[] loop_input;

//	List<float[][]> bptt_compute = new ArrayList<>();
	List<FloatMatrix> bptt_compute = new ArrayList<FloatMatrix>();
	//過去に誤差逆伝播で受け取った値
	List<FloatMatrix> bptt_dy = new ArrayList<>();
	List<FloatMatrix> past_input = new ArrayList<FloatMatrix>();
//	List<float[]> past_input = new ArrayList<>();
	//時刻t
	private int t;
	private static int ago = 3;

	/*
	 * リカレントの誤差計算のためのラベル
	 * 出力によって変更できるように設計するべきかどうか
	 */
//	train_T_minibatch = new int[minibatch_N][minibatchSize][nOut];
	FloatMatrix label;
	FloatMatrix input_data;
	FloatMatrix output_data;
//	float[][] label;



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


	public RNNlayer(int input, int output, FloatMatrix W, FloatMatrix r_w, FloatMatrix b, String actfunc,int minibatch, Sfmt m){
		if(m == null){
			int[] init_key = {(int) System.currentTimeMillis(), (int) Runtime.getRuntime().freeMemory()};
			m = new Sfmt(init_key);
		}
		//mt = m;

		//重みの初期化
		if(W == null){
			W = new FloatMatrix(output,input);
			float min = 1.f/ input;

			for(int i=0; i<output; i++){
				for(int j=0; j<input; j++){
					W.put(i,j, RandomGenerator.uniform(-min, min, m));
				}
			}
		}
		if(r_w == null){
			r_w = new FloatMatrix(input,input);
			float min = 1.f/ input;

			for(int i=0; i<input; i++){
				for(int j=0; j<input; j++){
					r_w.put(i,j, RandomGenerator.uniform(-min, min, m));
				}
			}
		}

		//バイアスの初期化
		if(b == null){
			b = new FloatMatrix(output);
		}

		input_N= input;
		output_N=output;

		bias=b;
		mt=m;
		selfloop_output = input;
		t = 0;

		weight=W;
		reccurent_weight = r_w;
		input_data = new FloatMatrix(minibatch, input);
		loop_input = new FloatMatrix(minibatch, input);
		output_data = new FloatMatrix(minibatch, output);

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

	/**
	 * 順伝播
	 * @param x 入力データ
	 * @return 出力
	 */
	public FloatMatrix forward(FloatMatrix x) {
		// TODO 自動生成されたメソッド・スタブ
		return output(x);
	}

	public FloatMatrix output(FloatMatrix x) {
		// TODO 自動生成されたメソッド・スタブ
		FloatMatrix y = new FloatMatrix(x.rows,output_N);
		//loop_inputを活性化した入力値
		FloatMatrix activate_loop = new FloatMatrix(x.rows,input_N);
		if(t != 0 && x.rows >= 0){
			//ループの入力に重みをかける
			activate_loop = loop_input.mmul(reccurent_weight);
			//biasを足したい場合はここで+recurrent_bias
		}
		//y[i] += weight[i][j] * x[j];
		//一番最初のフィードフォワードの場合
		FloatMatrix act = new FloatMatrix(x.rows, output_N);
		if(t == 0){
			act = x.mmul(weight.transpose());
//			activate_loop = x.dup();
		}else{
//			System.out.println(x.rows+"*"+x.columns+"#"+activate_loop.rows+"*"+activate_loop.columns);
			act = x.add(activate_loop).mmul(weight.transpose());
//			activate_loop[j] = act;
		}
		act.addiRowVector(bias);
		y = activation.apply(act);
		//誤差逆伝播に使用
		past_input.add(activate_loop.dup());
		if(past_input.size() >= (ago)*loop_input.length ){
			past_input.remove(0);
		}
		//次のリカレントの入力に使用
		loop_input = activate_loop.dup();
		t++;

//		System.out.println(loop_input.length);
		return y;
	}

	public FloatMatrix outputBinomial(FloatMatrix x, Sfmt mt) {
//		public int[] outputBinomial(int[] x, Sfmt mt) {

		FloatMatrix y = new FloatMatrix(output_N);

		FloatMatrix xCast = x.dup();
//		int[] y = new int[output_N];
//
//		float[] xCast = new float[x.length];
//		for (int i = 0; i < xCast.length; i++) {
//			xCast[i] = (float) x[i];
//		}

		FloatMatrix out = output(xCast);

		y = RandomGenerator.binomial(1, out, mt);
//		for (int j = 0; j < output_N; j++) {
//			y[j] = RandomGenerator.binomial(1, out[j], mt);
//		}

		return y;
	}

	/**
	 * 逆伝播メソッド
	 * 順伝播の重みを更新する。リカレントの重みはここではしない
	 * @param x この層への入力
	 * @param z この層の出力
	 * @param dy 前の層（出力に近い層）の逆伝播の値(誤差)
	 * @param minibatchSize ミニバッチサイズ
	 * @param l_rate 学習率
	 * @return 逆伝播の結果
	 */
	public FloatMatrix backward(FloatMatrix x, FloatMatrix weight_prev, FloatMatrix dy, int minibatchSize, float l_rate) {
		// TODO 自動生成されたメソッド・スタブ
		//逆伝播する誤差
//		float[][] dz = new float[minibatchSize][output_N];
		//さかのぼる時刻の設定
		return BPTT(minibatchSize, dy, weight_prev, l_rate);
	}

	/**
	 * Real Time Recurrent Learning
	 * @param minibatch
	 * @return
	 */
	public FloatMatrix RTRL(int minibatch, FloatMatrix x, FloatMatrix dy, float l_rate){
		//逆伝播する誤差
		FloatMatrix dz = new FloatMatrix(minibatch,output_N);
		return dz;
	}

	/**
	 * BackPropagation Through Time
	 * @param minibatch ミニバッチサイズ
	 * @param dy 前の層の誤差
	 * @param weight_perev 前の層の重み、こいつが出力層の場合はこいつの重みを渡すこと
	 * @param l_rate 学習率
	 * @return 順伝播用の誤差
	 */
	private FloatMatrix BPTT(int minibatch, FloatMatrix dy, FloatMatrix weight_prev, float l_rate){
		//現在の記憶している誤差
		int step = bptt_compute.size();
		//逆伝播する誤差
//		System.out.println(bptt_dy.size()+":"+step);
		FloatMatrix dz = new FloatMatrix(minibatch,input_N);
		FloatMatrix[] dz_step = new FloatMatrix[step+1];//[minibatch][input_N];

		FloatMatrix[] past_dz = new FloatMatrix[step+1];//[minibatch][input_N];

		//次の層からの逆伝播
		FloatMatrix[] past_dy = new FloatMatrix[step+1];//[minibatch][input_N];

		//式7.7の第一項、現時点の誤差を計算
		dz = dy.mmul(weight_prev);
//		for(int m=0; m<minibatch; m++)
//			for(int i=0; i< input_N; i++)
//				for(int p=0; p<dy[0].length; p++){
//					dz[m][i] += weight_prev[p][i] * dy[m][p];
//				}

		//第1項の過去分の計算
		for(int s=step-1; s>=0; s--){
			//					System.out.println(s+":"+m+":"+i+":"+step);
			//						step == 1:過去の履歴がない場合, ago-1:t+1の場合誤差がないため
			if(s == ago || step == 0){
				//System.out.println(past_dz.length);//+"*"+past_dz[0].length+"*"+past_dz[0][0].length);
				past_dz[s] = dz.dup();
			}else{
				past_dz[s] = bptt_dy.get(s).mmul(weight_prev);// * reccurent_weight[i][i] * dactivation.apply(past_input.get(s)[i]);
				//						past_dz[s][m][i] = dz_step[s+1][m][i] * reccurent_weight[i][i] * dactivation.apply(past_input.get(s)[i]);
			}
		}
//		for(int s=step-1; s>=0; s--){
//			for(int m=0; m<minibatch; m++){
//				for(int i=0; i< input_N; i++){
//					for(int j=0; j<dy[0].length; j++){
////					System.out.println(s+":"+m+":"+i+":"+step);
////						step == 1:過去の履歴がない場合, ago-1:t+1の場合誤差がないため
//						if(s == ago || step == 0){
//							//System.out.println(past_dz.length);//+"*"+past_dz[0].length+"*"+past_dz[0][0].length);
//							past_dz[s] = dz.clone();
//						}else{
//							past_dz[s][m][i] = bptt_dy.get(s)[m][j] * weight_prev[j][i];// * reccurent_weight[i][i] * dactivation.apply(past_input.get(s)[i]);
////						past_dz[s][m][i] = dz_step[s+1][m][i] * reccurent_weight[i][i] * dactivation.apply(past_input.get(s)[i]);
//						}
//					}
//				}
//			}
//		}

		//第2項はこのリカレント層のデルタは、出力と正解ラベルの誤差関数の微分
		//つまり問題によって誤差関数が変わる。
		//https://micin.jp/feed/developer/articles/rnn000 の22式より
		for(int s=step-1; s>=0; s--){
			if(s == ago || step == 0){
				past_dy[s].mul(0);
			}else{
				//						System.out.println(i+"#"+input_N+"#"+bptt_compute.size()+"#"+bptt_compute.get(s).length+"#"+bptt_compute.get(s)[m].length);
				past_dy[s] = bptt_compute.get(s).mmul(reccurent_weight);// * dactivation.apply(past_input.get(s)[i]);
			}
			//デルタを計算
			dz_step[s] = dactivation.apply(past_input.get(s)).mul((past_dy[s].add(past_dz[s])));
		}
//		for(int s=step-1; s>=0; s--){
//			for(int m=0; m<minibatch; m++){
//				for(int i=0; i< input_N; i++){
//					if(s == ago || step == 0){
//						past_dy[s][m][i] = 0;
//					}else{
////						System.out.println(i+"#"+input_N+"#"+bptt_compute.size()+"#"+bptt_compute.get(s).length+"#"+bptt_compute.get(s)[m].length);
//						past_dy[s][m][i] = bptt_compute.get(s)[m][i] * reccurent_weight[i][i];// * dactivation.apply(past_input.get(s)[i]);
//					}
//					//デルタを計算
//					dz_step[s][m][i] = (past_dy[s][m][i] + past_dz[s][m][i]) * dactivation.apply(past_input.get(s*minibatch+m)[i]);
//				}
//			}
//		}

		FloatMatrix grad_weight = new FloatMatrix(input_N,input_N);
		for(int s=0; s<step; s++){
						grad_weight.addi( dz_step[s].transpose().mmul(past_input.get(s)));
		}
//		for(int s=0; s<step; s++)
//			for(int m=0; m<minibatch; m++)
//				for(int i=0; i<input_N; i++)
//					for(int j=0; j<input_N; j++){
//						grad_weight[i][j] += dz_step[s][m][i] * past_input.get(s*minibatch+m)[i];
//					}

		//リカレントの重みを更新
		reccurent_weight.subi(grad_weight.mul(l_rate).div(minibatch));
//		for(int i=0; i<input_N; i++)
//			for(int j=0; j<input_N; j++){
//				reccurent_weight[i][j] -= l_rate * grad_weight[i][j] / minibatch;
//			}

		//順伝播重み誤差
		FloatMatrix grad_w = new FloatMatrix(output_N,input_N);
		FloatMatrix grad_b = new FloatMatrix(output_N);//バイアス

		grad_w = dy.transpose().mmul(loop_input);

//		for(int m=0; m<minibatch; m++)
//			for(int i=0; i<output_N; i++)
//				for(int j=0; j<input_N; j++){
//					grad_w[i][j] += dy[m][i] * loop_input[m][j];
//					grad_b[i] += dz[m][j];
//				}
		//バイアスの更新
		//重みとバイアスを更新
		weight.subi(grad_w.mul(l_rate).div(minibatch));
		bias.subi(grad_b.mul(l_rate).div(minibatch));
//		for(int i=0; i<output_N; i++){
//			for(int j=0; j<input_N; j++){
//				weight[i][j] -= l_rate * grad_w[i][j] / minibatch;
//			}
//			bias[i] -= l_rate * grad_b[i] / minibatch;
//		}

		//歴代誤差を記憶
		if(bptt_compute.size() >= ago){
			bptt_compute.remove(0);
			bptt_dy.remove(0);
		}
//		System.out.println(dz.length+"+"+dz[0].length);
		bptt_compute.add(dz.dup());
		bptt_dy.add(dy.dup());
//		System.out.println(bptt_compute.size()+"#"+bptt_compute.get(bptt_compute.size()-1).length+"#"+bptt_compute.get(bptt_compute.size()-1)[0].length);
		//一番最新の誤差を返す
		return dz;
	}

	public FloatMatrix train(FloatMatrix x, FloatMatrix label, int minibatch, float l_rate){
		FloatMatrix dy = output(x).sub(label);
//		for(int m=0; m<minibatch; m++){
//			float[] y = output(x[m],m);
//			for(int i=0; i<output_N; i++){
//				dy[m][i] = y[i] - (float) label[m][i];
//			}
//		}

		return BPTT(minibatch,dy,weight,l_rate);
	}




	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ
		//RNNlayer(int input, int output, float[][] W, float[][] r_w, float[] b, String actfunc, Sfmt m)
		int[] init_key = {(int) System.currentTimeMillis(), (int) Runtime.getRuntime().freeMemory()};
		Sfmt mt = new Sfmt(init_key);

		int minibatch = 20;

		RNNlayer rnn = new RNNlayer(5, 3, null, null, null, "ReLU", minibatch,mt);

		FloatMatrix x = FloatMatrix.randn(minibatch, rnn.input_N);
		FloatMatrix z = new FloatMatrix(minibatch,rnn.output_N);
		FloatMatrix dy = new FloatMatrix(minibatch,rnn.output_N);
		FloatMatrix label = FloatMatrix.rand(minibatch, rnn.output_N).ge(0.5f);

		for(int epoch =0; epoch < 10; epoch++){
				z = rnn.forward(x);
		}

		FloatMatrix error = rnn.backward(x, rnn.weight, FloatMatrix.randn(minibatch, rnn.output_N), minibatch, 0.01f);


		System.out.printf("error[%d][%d]\n", error.rows, error.columns );
		System.out.println("error:"+ error);


		System.out.println("\nrnn test end");
	}

}
