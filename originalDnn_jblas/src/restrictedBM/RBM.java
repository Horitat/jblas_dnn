package restrictedBM;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.jblas.FloatMatrix;

import util.ActivationFunction;
import util.Common_method;
import util.RandomGenerator;
import Mersenne.Sfmt;

public class RBM {

	public int input_N;
	public int hidden_N;
	public FloatMatrix weight;
	//	public float[][] weight;
	public FloatMatrix hbias;
	public FloatMatrix ibias;
	//	public float[] hbias;
	//	public float[] ibias;
	public Sfmt mt;


	public RBM(int input, int hidden, FloatMatrix w, FloatMatrix hb, FloatMatrix ib, Sfmt m){
		//		public RBM(int input, int hidden, float[][] w, float[] hb, float[] ib, Sfmt m){
		if(m == null){
			int[] init_key = {(int) System.currentTimeMillis(), (int) Runtime.getRuntime().freeMemory()};
			m = new Sfmt(init_key);
		}

		if(w == null){
			//隠れ層と入出力層のサイズの二次元配列
			w = new FloatMatrix(hidden,input);
			float w_ = 1.f / input;

			for(int i=0; i<hidden; i++){
				for(int j=0; j<input; j++){
					//重みの初期化
					w.put(i,j,RandomGenerator.uniform(-w_, w_, m));
				}
			}
		}

		if(hb == null){
			hb = FloatMatrix.zeros(hidden).transpose();
			//			hb = new float[hidden];
			//			for(int i=0; i<hidden; i++){
			//				hb[i] = 0.f;
			//			}
		}

		if(ib == null){
			ib = FloatMatrix.zeros(input).transpose();
			//			ib = new float[input];
			//			for(int i=0; i<input; i++){
			//				ib[i] = 0.f;
			//			}
		}
		System.out.println("Creat RBM:" + input +":" + hidden);
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

		int train_N_each = 200;         // for demo
		int test_N_each = 2;            // for demo
		int nVisible_each = 4;          // for demo
		double pNoise_Training = 0.05;  // for demo
		double pNoise_Test = 0.25;      // for demo

		final int patterns = 3;

		final int train_N = train_N_each * patterns;
		final int test_N = test_N_each * patterns;

		final int nVisible = nVisible_each * patterns;//入力層の数
		int nHidden = 6;//出力層の数

		FloatMatrix train_X = new FloatMatrix(train_N,nVisible);
		FloatMatrix test_X = new FloatMatrix(test_N,nVisible);


		//トレーニングデータの生成
		for (int pattern = 0; pattern < patterns; pattern++) {
			for (int n = 0; n < train_N_each; n++) {

				int n_ = pattern * train_N_each + n;

				for (int i = 0; i < nVisible; i++) {
					if ( (n_ >= train_N_each * pattern && n_ < train_N_each * (pattern + 1) ) &&
							(i >= nVisible_each * pattern && i < nVisible_each * (pattern + 1)) ) {
						train_X.put(n_,i,RandomGenerator.binomial(1, 1-pNoise_Training, mt));
					} else {
						train_X.put(n_,i,RandomGenerator.binomial(1, pNoise_Training, mt));
					}
				}
			}

			for (int n = 0; n < test_N_each; n++) {

				int n_ = pattern * test_N_each + n;

				for (int i = 0; i < nVisible; i++) {
					if ( (n_ >= test_N_each * pattern && n_ < test_N_each * (pattern + 1) ) &&
							(i >= nVisible_each * pattern && i < nVisible_each * (pattern + 1)) ) {
						test_X.put(n_, i, RandomGenerator.binomial(1, 1-pNoise_Test, mt));
					} else {
						test_X.put(n_,i,RandomGenerator.binomial(1, pNoise_Test, mt));
					}
				}
			}
		}

		//ミニバッチの精整
		int epochs = 1000;
		float l_rate = 0.2f;
		int minibatchSize = 10;
		final int minibatch_N = train_N / minibatchSize;

		FloatMatrix[] train_X_minibatch = new FloatMatrix[minibatch_N];//[minibatchSize][nVisible];
		List<Integer> minibatchIndex = new ArrayList<>();

		for (int i = 0; i < train_N; i++) {
			minibatchIndex.add(i);
		}
		Common_method.list_shuffle(minibatchIndex, mt);
		for (int i = 0; i < minibatch_N; i++) {
			train_X_minibatch[i] = new FloatMatrix(minibatchSize, nVisible);

			for (int j = 0; j < minibatchSize; j++) {
				train_X_minibatch[i].putRow(j, train_X.getRow(minibatchIndex.get(i * minibatchSize + j)));
			}
		}

		RBM nn = new RBM(nVisible, nHidden, null, null, null, mt);

		//コントラスティブダイバージェンスによる学習
		for(int epoch=0; epoch<epochs; epoch++){
			for(int batch=0; batch<minibatch_N; batch++){
				nn.contrastive(train_X_minibatch[batch],minibatchSize,l_rate,1);
			}
			l_rate *= 0.95;
		}

		FloatMatrix reconstructed_X = new FloatMatrix(test_N,nVisible);
		// test (reconstruct noised data)
		reconstructed_X = nn.reconstruct(test_X);

		// evaluation
		System.out.println("-----------------------------------");
		System.out.println("RBM model reconstruction evaluation");
		System.out.println("-----------------------------------");

		for (int pattern = 0; pattern < patterns; pattern++) {

			System.out.printf("Class%d\n", pattern + 1);
			//reconstructed_X = new float[test_N][nVisible=12];
			//test_N = test_N_each(2) * patterns(3);=6
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
		System.out.println("end");
	}

	/**
	 * パラメータ更新メソッド
	 * @param x トレーニングデータ
	 * @param minibatchSize ミニバッチサイズ
	 * @param l_rate 学習率
	 * @param k CD-KのKの回数
	 */
	public void contrastive(FloatMatrix x, int minibatchSize, float l_rate, int k) {
		//			public void contrastive(int[][] x, int minibatchSize, float l_rate, int k) {
		// TODO 自動生成されたメソッド・スタブ

		FloatMatrix phMean = new FloatMatrix(minibatchSize,hidden_N); //順伝播の隠れ層の活性
		FloatMatrix phSample = new FloatMatrix(minibatchSize,hidden_N); //順伝播の隠れ層のサンプリング
		FloatMatrix nvMean = new FloatMatrix(minibatchSize,input_N); //出力層からの入出力層への活性
		FloatMatrix nvSample = new FloatMatrix(minibatchSize,input_N); //出力層からの入出力層へのサンプリング
		FloatMatrix nhMean = new FloatMatrix(minibatchSize,hidden_N); //内部ループの出力層の活性
		FloatMatrix nhSample = new FloatMatrix(minibatchSize,hidden_N); //内部ループの出力層のサンプリング

		//			float[] phMean = new float[hidden_N]
		//			int[] phSample = new int[hidden_N];
		//			float[] nvMean = new float[input_N];
		//			int[] nvSample = new int[input_N];
		//			float[] nhMean = new float[input_N];
		//			int[] nhSample = new int[input_N];

		FloatMatrix grad_weight = new FloatMatrix(hidden_N,input_N);
		FloatMatrix grad_hbias = new FloatMatrix(hidden_N);
		FloatMatrix grad_ibias = new FloatMatrix(input_N);
		//			float grad_weight[][] = new float[hidden_N][input_N];
		//			float grad_hbias[] = new float[hidden_N];
		//			float grad_ibias[] = new float[input_N];

		sampleHgivenV(x, phMean, phSample);
//		System.out.println("phmean:"+phMean);
//		System.out.println("phsample:"+phSample);
		for(int step=0; step<k; step++){
			if(step==0){
				gibbsHVH(phSample,nvMean, nvSample, nhMean, nhSample);
			}else{
				gibbsHVH(nhSample, nvMean, nvSample, nhMean, nhSample);
			}
		}

		grad_weight = phMean.transpose().mmul(x).sub(nhMean.transpose().mmul(nvSample)).dup();
		grad_hbias = phMean.sub(nhMean).dup().columnSums();

		grad_ibias = x.sub(nvSample).dup().columnSums();
		/*			for(int n=0; n<minibatchSize; n++){
				sampleHgivenV(x[n], phMean, phSample);

				for(int step=0; step<k; step++){
					if(step==0){
						gibbsHVH(phSample,nvMean, nvSample, nhMean, nhSample);
					}else{
						gibbsHVH(nhSample, nvMean, nvSample, nhMean, nhSample);
					}
				}

				for(int i=0; i<hidden_N; i++){
					for(int j=0; j<input_N; j++){
						grad_weight[i][j] += phMean[i] * x[n][j] - nhMean[i] * nvSample[j];
					}
					grad_hbias[i] += phMean[i] - nhMean[i];
				}

				for(int i=0; i<input_N; i++){
					grad_ibias[i] += x[n][i] - nvSample[i];
				}
			}*/

		weight.addi(grad_weight.mul(l_rate).div(minibatchSize));
		//			System.out.println("hbias:"+hbias.rows+":"+hbias.columns);
		//			System.out.println("ghbias:"+grad_hbias.rows+":"+grad_hbias.columns);
		hbias.addi(grad_hbias.mul(l_rate).div(minibatchSize));
		ibias.addi(grad_ibias.mul(l_rate).div(minibatchSize));

		//			for(int i=0; i<hidden_N; i++){
		//				for(int j=0; j<input_N; j++){
		//					weight[i][j] += l_rate * grad_weight[i][j] / minibatchSize;
		//				}
		//				hbias[i] += l_rate * grad_hbias[i] / minibatchSize;
		//			}
		//
		//			for(int i=0; i<input_N; i++){
		//				ibias[i] += l_rate * grad_ibias[i] / minibatchSize;
		//			}
	}

	public FloatMatrix reconstruct(FloatMatrix v){
		//			public float[] reconstruct(int[] v){
		//			FloatMatrix x = new FloatMatrix(input_N);
		FloatMatrix h = propup(v, weight, hbias);
		//			float[] x = new float[input_N];
		//			float[] h = new float[hidden_N];

		//			for(int i=0; i<hidden_N; i++){
		//				h[i] = propup(v, weight[i], hbias[i]);
		//			}


		//			for(int i=0; i<input_N; i++){
		//				float pre = 0.f;
		//
		//				for(int j=0; j<hidden_N; j++){
		//					pre += weight[j][i] * h[j];
		//				}
		//				pre += ibias[i];
		//				x[i] = ActivationFunction.logistic_sigmoid(pre);
		//			}
//					System.out.println(h.rows+"--"+h.columns);
//					System.out.println(weight.rows+"--"+weight.columns);
//					System.out.println(weight.transpose().mmul(h).transpose());
					//System.out.println(h.mmul(weight));

//					System.out.println(ibias.rows+"ibias"+ibias.columns);
//		return ActivationFunction.logistic_sigmoid(weight.transpose().mmul(h).transpose().addRowVector(ibias));
		return ActivationFunction.logistic_sigmoid(h.mmul(weight).addRowVector(ibias));
	}

	/**
	 * ギブズサンプル
	 * @param phSample 隠れ層の入力
	 * @param nvMean 入出力層の活性
	 * @param nvSample 入出力層の入力（サンプリング）
	 * @param nhMean 隠れ層の活性
	 * @param nhSample 隠れ層の活性に基づくサンプリング
	 */
	public void gibbsHVH(FloatMatrix phSample, FloatMatrix nvMean, FloatMatrix nvSample, FloatMatrix nhMean, FloatMatrix nhSample) {
		//			public void gibbsHVH(int[] phSample, float[] nvMean, int[] nvSample, float[] nhMean, int[] nhSample) {
		// TODO 自動生成されたメソッド・スタブ
		sampleVgivenH(phSample, nvMean, nvSample);
		sampleHgivenV(nvSample, nhMean, nhSample);
	}

	/**
	 * 隠れ層の値をもとに入出力層で生成される確率分布とサンプリングデータを設定
	 * @param is 隠れ層の入力
	 * @param mean 入出力層の活性
	 * @param sample 活性に基づくサンプリング
	 */
	private void sampleVgivenH(FloatMatrix is, FloatMatrix mean, FloatMatrix sample) {
		//			private void sampleVgivenH(int[] is, float[] mean, int[] sample) {
		// TODO 自動生成されたメソッド・スタブ
		mean.muli(0);
		sample.muli(0);

		mean.addi(propdown(is, 0, ibias));
//		System.out.println("mean:"+mean);
		sample.addi(RandomGenerator.binomial(1, mean, mt));
//		System.out.println("sample:"+sample);
		//			for(int i=0; i<input_N; i++){
		//				mean[i] = propdown(is, i, ibias[i]);
		//				sample[i] = RandomGenerator.binomial(1, mean[i], mt);
		//			}
	}

	/**
	 * 入出力層の値をもとに隠れ層で生成される確率分布とサンプリングデータを設定
	 * @param is 入出力層の入力
	 * @param mean 隠れ層の活性
	 * @param sample 活性に基づくサンプリング
	 */
	public void sampleHgivenV(FloatMatrix is, FloatMatrix mean, FloatMatrix sample) {
		//			public void sampleHgivenV(int[] is, float[] mean, int[] sample) {
		// TODO 自動生成されたメソッド・スタブ
		mean.muli(0);
		sample.muli(0);

		mean.addi(propup(is, weight, hbias));
//		System.out.println("mean:"+mean);
		sample.addi(RandomGenerator.binomial(1, mean, mt));
//		System.out.println("sample:"+sample);
		//			for(int i=0; i<hidden_N; i++){
		//				mean[i] = propup(is, weight[i], hbias[i]);
		//				sample[i] = RandomGenerator.binomial(1, mean[i], mt);
		//			}

	}

	/**
	 * 入出力層から隠れ層への入力
	 * @param v 入力値
	 * @param w 重み
	 * @param bias バイアス値
	 * @return 隠れ層への入力値
	 */
	public FloatMatrix propup(FloatMatrix v, FloatMatrix w, FloatMatrix bias){
		//			public float propup(int[] v, float[] w, float bias){
		//			System.out.println(v.rows+"*"+v.columns);
		//			System.out.println(w.rows+"$$$"+w.columns);
		FloatMatrix input_to_hidden = v.mmul(weight.transpose());
		//			for(int i=0; i< input_N; i++){
		//				input_to_hidden += w[i] * v[i];
		//			}
		//			input_to_hidden += bias;
		//			System.out.println(input_to_hidden.rows+"++"+input_to_hidden.columns);
		//			System.out.println(hbias.rows+"hbias"+hbias.columns);
		return ActivationFunction.logistic_sigmoid(input_to_hidden.addRowVector(hbias));
//		return input_to_hidden;
	}

	/**
	 *隠れ層から入出力層への入力
	 * @param h 隠れ層からの出力値
	 * @param i 入出力層の番号
	 * @param bias バイアス値
	 * @return i番目の入出力層への入力値
	 */
	public FloatMatrix propdown(FloatMatrix h, int i, FloatMatrix bias){
		//			public float propdown(int[] h, int i, float bias){
//		System.out.println(h.rows+"--"+h.columns);
//		System.out.println(weight.rows+"++"+weight.columns);
//		System.out.println(weight.transpose().mmul(h).transpose());
//		FloatMatrix hidden_to_input = weight.transpose().mmul(h.transpose()).transpose();
		FloatMatrix hidden_to_input = h.mmul(weight);

		//			for(int j=0; j< hidden_N; j++){
		//				input_to_input += weight[j][i] * h[j];
		//			}
		//preActivation += bias;
//		System.out.println(hidden_to_input.rows+"...."+hidden_to_input.columns);
//		System.out.println(ibias.rows+"^^^"+ibias.columns);
		return ActivationFunction.logistic_sigmoid(hidden_to_input.addRowVector(ibias));
	}
}
