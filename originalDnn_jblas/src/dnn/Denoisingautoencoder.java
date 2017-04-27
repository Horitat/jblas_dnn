package dnn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.jblas.FloatMatrix;

import util.ActivationFunction;
import util.Common_method;
import util.RandomGenerator;
import Mersenne.Sfmt;

public class Denoisingautoencoder {

	int input_N;
	int hidden_N;
	FloatMatrix weight;
	FloatMatrix hbias;
	FloatMatrix ibias;
//	float[][] weight;
//	float[] hbias;
//	float[] ibias;
	Sfmt mt;

	public Denoisingautoencoder(int input, int hidden, FloatMatrix w, FloatMatrix hb, FloatMatrix ib, Sfmt m){
//		public Denoisingautoencoder(int input, int hidden, float[][] w, float[] hb, float[] ib, Sfmt m){
		if(m == null){
			int[] init_key = {(int) System.currentTimeMillis(), (int) Runtime.getRuntime().freeMemory()};
			m = new Sfmt(init_key);
		}

		if(w == null){
			//隠れ層と入出力層のサイズの二次元配列
			w = new FloatMatrix(hidden,input);
//			w = new float[hidden][input];
			float w_ = 1.f / input;

			for(int i=0; i<hidden; i++){
				for(int j=0; j<input; j++){
					//重みの初期化
					w.put(i,j, RandomGenerator.uniform(-w_, w_, m));
				}
			}
		}

		if(hb == null){
			hb = new FloatMatrix(1,hidden);
//			hb = new float[hidden];
//			for(int i=0; i<hidden; i++){
//				hb[i] = 0.f;
//			}
		}

		if(ib == null){
			ib = new FloatMatrix(1,input);
//			ib = new float[input];
//			for(int i=0; i<input; i++){
//				ib[i] = 0.f;
//			}
		}

		input_N = input;
		hidden_N = hidden;
		weight = w.dup();
		hbias = hb.dup();
		ibias = ib.dup();
		mt = m;
	}

	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ
		int[] init_key = {(int) System.currentTimeMillis(), (int) Runtime.getRuntime().freeMemory()};
		Sfmt mt = new Sfmt(init_key);

		float noise_lvl = 0.3f; //0.1~0.3

		int train_N_each = 200;         // for demo
		int test_N_each = 2;            // for demo
		int nVisible_each = 4;          // for demo
		float pNoise_Training = 0.05f;  // for demo
		float pNoise_Test = 0.25f;      // for demo

		final int patterns = 3;

		final int train_N = train_N_each * patterns;
		final int test_N = test_N_each * patterns;

		final int nVisible = nVisible_each * patterns;//入力層の数
		int nHidden = 6;//出力層の数

		FloatMatrix train_X = new FloatMatrix(train_N,nVisible);
		FloatMatrix test_X = new FloatMatrix(test_N,nVisible);
		FloatMatrix reconstructed_X = new FloatMatrix(test_N,nVisible);

		int epochs = 1000;
		float l_rate = 0.2f;
		int minibatchSize = 10;
		final int minibatch_N = train_N / minibatchSize;



		//トレーニングデータの生成
		for (int pattern = 0; pattern < patterns; pattern++) {
			for (int n = 0; n < train_N_each; n++) {
				int n_ = pattern * train_N_each + n;

				for (int i = 0; i < nVisible; i++) {
					if ( (n_ >= train_N_each * pattern && n_ < train_N_each * (pattern + 1) ) &&
							(i >= nVisible_each * pattern && i < nVisible_each * (pattern + 1)) ) {
						train_X.put(n_,i, RandomGenerator.binomial(1, 1-pNoise_Training, mt));
					} else {
						train_X.put(n_,i, RandomGenerator.binomial(1, pNoise_Training, mt));
					}
				}
			}

			for (int n = 0; n < test_N_each; n++) {
				int n_ = pattern * test_N_each + n;

				for (int i = 0; i < nVisible; i++) {
					if ( (n_ >= test_N_each * pattern && n_ < test_N_each * (pattern + 1) ) &&
							(i >= nVisible_each * pattern && i < nVisible_each * (pattern + 1)) ) {
						test_X.put(n_,i,RandomGenerator.binomial(1, 1-pNoise_Test, mt));
					} else {
						test_X.put(n_,i, RandomGenerator.binomial(1, pNoise_Test, mt));
					}
				}
			}
		}

		FloatMatrix[] train_X_minibatch = new FloatMatrix[minibatch_N];//[minibatchSize][nVisible];
		List<Integer> minibatchIndex = new ArrayList<>();

		for (int i = 0; i < train_N; i++) {
			minibatchIndex.add(i);
		}
		Common_method.list_shuffle(minibatchIndex, mt);
		//ミニバッチの精整
		for (int i = 0; i < minibatch_N; i++) {
			train_X_minibatch[i] = new FloatMatrix(minibatchSize,nVisible);
			for (int j = 0; j < minibatchSize; j++) {
				train_X_minibatch[i].putRow(j, train_X.getRow(minibatchIndex.get(i * minibatchSize + j)));
//				System.out.println(train_X_minibatch[i].getRow(j));
//				System.out.println(train_X.getRow(minibatchIndex.get(i * minibatchSize + j)));
			}
//			System.out.println("                      ");
		}

		Denoisingautoencoder nn = new Denoisingautoencoder(nVisible, nHidden, null, null, null, mt);
		//training
		for(int epoch=0; epoch<epochs; epoch++){
			for(int batch=0; batch<minibatch_N; batch++){
				nn.train(train_X_minibatch[batch], minibatchSize, l_rate, noise_lvl);
			}
			System.out.println(epoch +"/"+epochs);
		}

		reconstructed_X = nn.reconstruct(test_X);

		// evaluation
		System.out.println("-----------------------------------");
		System.out.println("DA model reconstruction evaluation");
		System.out.println("-----------------------------------");

		for (int pattern = 0; pattern < patterns; pattern++) {
			System.out.printf("Class%d\n", pattern + 1);
			for (int n = 0; n < test_N_each; n++) {
				int n_ = pattern * test_N_each + n;
				System.out.print( Arrays.toString(test_X.getRow(n_).toIntArray()) + " -> ");
				System.out.print("[");
				for (int i = 0; i < nVisible-1; i++) {
					System.out.printf("%.5f, ", reconstructed_X.get(n_,i));
				}
				System.out.printf("%.5f]\n", reconstructed_X.get(n_,nVisible-1));
			}
			System.out.println();
		}

	}

	/**
	 * トレーニングメソッド
	 * @param x 入力データ
	 * @param minibatchSize ミニバッチのサイズ
	 * @param l_rate 学習率
	 * @param noise_lvl 入力データにノイズを加える確率
	 */
	public void train(FloatMatrix x, int minibatchSize, float l_rate, float noise_lvl) {
//		public void train(float[][] x, int minibatchSize, float l_rate, float noise_lvl) {
		// TODO 自動生成されたメソッド・スタブ
		FloatMatrix grad_weight = new FloatMatrix(hidden_N,input_N);
		FloatMatrix grad_hbias = new FloatMatrix(1,hidden_N);
		FloatMatrix grad_ibias = new FloatMatrix(1,input_N);
//		float[][] grad_weight = new float[hidden_N][input_N];
//		float[] grad_hbias = new float[hidden_N];
//		float[] grad_ibias = new float[input_N];

		FloatMatrix noise_data = getNoisedata(x.dup(), noise_lvl);
//		System.out.println("n:"+noise_data);
//		System.out.println("x:"+x);

		//隠れ層へ転送。エンコード
		FloatMatrix z = caluculation_hidden(noise_data.dup());
//		System.out.println("Z:"+z);
		//出力層へ転送。デコード
		FloatMatrix y = caluculation_output(z.dup());
//		System.out.println("Y1:"+y);
//		System.out.println("Y2:"+caluculation_output(z.dup()));
		//勾配を計算
//		System.out.println("gww:"+grad_weight);
//		System.out.println("gii"+grad_ibias);
//		System.out.println("ghh:"+grad_hbias);

//		System.out.println("X:"+x);
//		System.out.println("Z:"+z);
//		System.out.println("Y:"+y);
		caluculation_grad(grad_ibias, grad_hbias, x, y, grad_weight, z, noise_data);

//		System.out.println("x:"+x);
//		System.out.println("z:"+z);
//		System.out.println("y:"+y);

//		System.out.println("gw:"+grad_weight);
//		System.out.println("gi"+grad_ibias);
//		System.out.println("gh:"+grad_hbias);

//		for(int n=0; n<minibatchSize; n++){
//			float[] noise_data = getNoisedata(x[n], noise_lvl);
//			//隠れ層へ転送。エンコード
//			float[] z = caluculation_hidden(noise_data);
//			//出力層へ転送。デコード
//			float[] y = caluculation_output(z);
//			//勾配を計算
//			caluculation_grad(grad_ibias, grad_hbias, x, y, grad_weight, z, noise_data, n);
//		}

//		System.out.println("w:"+weight);

//		System.out.println("gww:"+grad_weight);
//		System.out.println("gii"+grad_ibias);
//		System.out.println("ghh:"+grad_hbias);
		//重み、バイアスをアップデート
		update_param(grad_ibias, grad_hbias, grad_weight, l_rate, minibatchSize);
	}

	/**
	 * 勾配の計算
	 * @param grad_ibias 入力層の勾配
	 * @param grad_hbias 隠れ層の勾配
	 * @param x 元データ
	 * @param y 出力
	 * @param grad_weight 重みの勾配
	 * @param z 隠れ層の出力
	 * @param noise_data 入力データ
	 */
	private void caluculation_grad(FloatMatrix grad_ibias,FloatMatrix grad_hbias,
			FloatMatrix x, FloatMatrix y, FloatMatrix grad_weight, FloatMatrix z, FloatMatrix noise_data) {
//		private void caluculation_grad(float[] grad_ibias, float[] grad_hbias,
//				float[][] x, float[] y, float[][] grad_weight, float[] z, float[] noise_data, int n) {
		// TODO 自動生成されたメソッド・スタブ
		//input bias grad

//		float[] io = new float[input_N];
//		for(int i=0; i< input_N; i++){
//			io[i] = x[n][i] - y[i];
//			grad_ibias[i] += io[i];
//		}
//		System.out.println(x.rows+"xxx"+x.columns);
//		System.out.println(y.rows+"yyy"+y.columns);
//		System.out.println("xxx"+x);
//		System.out.println("yyy"+y);
		FloatMatrix io = x.sub(y);
//		System.out.println("i"+io.columnSums());
//		System.out.println(io.rows+"i"+io.columns);
//		System.out.println(":i:"+io);
//		System.out.println(io.columnSums().rows+":i:"+io.columnSums().columns);
//		System.out.println("i"+grad_ibias);
//		System.out.println("i"+grad_ibias.rows+"**"+grad_ibias.columns);
		grad_ibias.addi(io.columnSums());
//		System.out.println("i"+grad_ibias);

		//hidden bias grad
		FloatMatrix h = io.mmul(weight.transpose());
//		System.out.println("i"+z);
		h.muli(z.mul(z.dup().rsub(1)));
//		System.out.println("h"+h.columnSums());
//		System.out.println(h.rows+"h"+h.columns);
//		System.out.println(h.columnSums().rows+":h:"+h.columnSums().columns);
//		System.out.println("h"+grad_hbias.rows+"**"+grad_hbias.columns);
		grad_hbias.addi(h.columnSums());
//		System.out.println("h"+grad_hbias);
//		System.out.println("p"+z);
//		float[] h = new float[hidden_N];
//		for(int i=0; i<hidden_N; i++){
//			for(int j=0; j<input_N; j++){
//				h[i] += weight[i][j] *(x[n][j] - y[j]);
//			}
//			h[i] *= ActivationFunction.dsigmoid(z[i]);
//			grad_hbias[i] += h[i];
//		}

		//weight grad
//		System.out.println("W"+grad_weight);
//		FloatMatrix g = h.transpose().mmul(noise_data);
//		FloatMatrix m = z.transpose().mmul(io);
//		FloatMatrix o = g.add(m);
//		System.out.println(m.rows+"m"+m.columns);
//		System.out.println(g.rows+"g"+g.columns);
//		grad_weight.addi(o);
		grad_weight.addi(h.transpose().mmul(noise_data).add(z.transpose().mmul(io)));
//		System.out.println("Ww"+grad_weight);
//		for(int i=0; i<hidden_N; i++){
//			for(int j=0; j<input_N; j++){
//				grad_weight[i][j] += h[i] * noise_data[j] + io[j] * z[i];
//			}
//		}
	}

	/**
	 * パラメータアップデート
	 * @param grad_ibias 入力層バイアスの勾配
	 * @param grad_hbias 出力層バイアスの勾配
	 * @param grad_weight 重みの勾配
	 * @param l_rate 学習率
	 * @param minibatchSize ミニバッチサイズ
	 */
	private void update_param(FloatMatrix grad_ibias, FloatMatrix grad_hbias, FloatMatrix grad_weight,
			float l_rate, int minibatchSize) {
//		private void update_param(float[] grad_ibias, float[] grad_hbias, float[][] grad_weight,
//				float l_rate, int minibatchSize) {
		// TODO 自動生成されたメソッド・スタブ

//		System.out.println("gw:"+grad_weight);
//		System.out.println("gw:"+grad_weight.mul(l_rate).div((float)minibatchSize));
//		System.out.println("gi"+grad_ibias);
//		System.out.println("gh"+grad_hbias);


//		System.out.println(weight.rows+":ww:"+weight.columns);
//		System.out.println(grad_weight.rows+":gw:"+grad_weight.columns);
//		System.out.println(ibias.rows+":ib:"+ibias.columns);
//		System.out.println(grad_ibias.rows+"gi"+grad_ibias.columns);
//		System.out.println(hbias.rows+":hb:"+hbias.columns);
//		System.out.println(grad_hbias.rows+"gh"+grad_hbias.columns);

//		FloatMatrix delta = grad_weight.mul(l_rate);
//		delta.divi((float)minibatchSize);
//		weight.addi(delta);
		weight.addi(grad_weight.mul(l_rate).div((float)minibatchSize));
		hbias.addi(grad_hbias.mul(l_rate).div((float)minibatchSize));
		ibias.addi(grad_ibias.mul(l_rate).div((float)minibatchSize));

//		for(int i=0; i<hidden_N; i++){
//			for(int j=0; j<input_N; j++){
//				weight[i][j] += l_rate * grad_weight[i][j] / minibatchSize;
//			}
//			hbias[i] += l_rate * grad_hbias[i] / minibatchSize;
//		}

//		for(int i=0; i<input_N; i++){
//			ibias[i] += l_rate * grad_ibias[i] / minibatchSize;
//		}
	}

	/**
	 * 入力データにノイズを加える
	 * @param x 入力データ
	 * @param noise_lvl ノイズを加える確率
	 * @return ノイズを加えた入力データ
	 */
	private FloatMatrix getNoisedata(FloatMatrix x, float noise_lvl) {
//		private float[] getNoisedata(float[] x, float noise_lvl) {
		// TODO 自動生成されたメソッド・スタブ
//		FloatMatrix noise_data = FloatMatrix.zeros(x.rows, x.columns);
		FloatMatrix noise_data = x.dup();
		//noise_lvl未満の乱数が出たら0
		for(int i=0; i<x.rows; i++){
			for(int j=0; j<x.columns;j++){
//				if(mt.NextUnif() >= noise_lvl){
//					noise_data.put(i, j, x.get(i,j));
//				}
				if(mt.NextUnif() < noise_lvl){
					noise_data.put(i, j, 0.f);
				}
			}
		}
//		System.out.println("n:"+noise_data);
//		System.out.println("x:"+x);
//		System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
//		for(int i=0; i<x.length; i++){
//			if(mt.NextUnif() < noise_lvl){
//				noise_data[i] = 0;
//			}else{
//				noise_data[i] = x[i];
//			}
//		}
		return noise_data;
	}

	/**
	 * テストを行うメソッド
	 * @param x テストデータ
	 * @return テストデータの計算結果
	 */
	public FloatMatrix reconstruct(FloatMatrix x){
//		public float[] reconstruct(float[] x){
		return caluculation_output(caluculation_hidden(x));
	}

	/**
	 * 隠れ層の出力を返す
	 * @param noise_data 入力データ
	 * @return 隠れ層の出力
	 */
	private FloatMatrix caluculation_hidden(FloatMatrix noise_data) {
//		private float[] caluculation_hidden(float[] noise_data) {
		// TODO 自動生成されたメソッド・スタブ
//		FloatMatrix z = new FloatMatrix(noise_data.rows,hidden_N);
//		float[] z = new float[hidden_N];
//		System.out.println("z:"+z);
//		System.out.println("Zi"+z);
//		for(int i=0; i<hidden_N; i++){
//			for(int j=0; j<input_N; j++){
//				z[i] += weight[i][j]*noise_data[j];
//			}
//			z[i] = ActivationFunction.logistic_sigmoid(z[i] + hbias[i]);
//		}
//		System.out.println(weight);
//		return ActivationFunction.logistic_sigmoid(z);
		return ActivationFunction.logistic_sigmoid(noise_data.mmul(weight.transpose()).addRowVector(hbias));
	}

	/**
	 * 出力層の計算結果を返す
	 * @param z 出力層への入力
	 * @return 計算結果
	 */
	private FloatMatrix caluculation_output(FloatMatrix z) {
//		private float[] caluculation_output(float[] z) {
		// TODO 自動生成されたメソッド・スタブ
//		FloatMatrix y = z.mmul(weight);
//		float[] y = new float[input_N];
//		for(int i=0; i<input_N; i++){
//			for(int j=0; j<hidden_N; j++){
//				y[i] += weight[j][i] * z[j];
//			}
//			y[i] = ActivationFunction.logistic_sigmoid(y[i] + ibias[i]);
//		}
//		return ActivationFunction.logistic_sigmoid(y);
		return ActivationFunction.logistic_sigmoid(z.mmul(weight).addRowVector(ibias));
	}
}
