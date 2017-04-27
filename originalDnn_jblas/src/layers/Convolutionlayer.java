package layers;

import java.util.Scanner;

import org.apache.commons.lang3.StringUtils;
import org.jblas.FloatMatrix;

import util.ActivationFunction;
import util.Jblas_util;
import util.RandomGenerator;
import Mersenne.Sfmt;

//import org.nd4j.linalg.api.ndarray.INDArray;
//import org.nd4j.linalg.factory.Nd4j;


public class Convolutionlayer extends Layer{

	int[] imagesize;
	//	float[][][][] weight;
	FloatMatrix[][] weight;
	float[][][][] adagrad_gradient;
	FloatMatrix bias;
	//	float[] bias;
	int chanel;
	int[] kernelsize;
	int[] convoutsize;
	int kernelnum;
	Sfmt mt;
	ActivationFunction.FloatMatrixFunction<FloatMatrix> activation;
	ActivationFunction.FloatMatrixFunction<FloatMatrix> dactivation;
	//	Hiddenlayer.FloatFunction<Float> activation;
	//	Hiddenlayer.FloatFunction<Float> dactivation;
	int flatsize;
	int stride;
	int padding;

	//モーメンタム、momentum*前のgrad_weightを現時点の重みの更新量に足す
	float momentum;
	//重みの減衰、l_rate*w_decay*weightを現時点の重みの更新量から引く
	float w_decay;


	FloatMatrix[][] input_data;
	FloatMatrix[][] preact_data;

	/**
	 * 畳込み層のコンストラクタ
	 * @param imgsize 入力画像サイズ
	 * @param chnl チャネル数
	 * @param nkernel カーネルの数
	 * @param kernelsize 畳込み層のカーネルサイズ
	 * @param convoutsize 畳込み層の出力数
	 * @param m メルセンヌツイスタ
	 * @param actfunc 活性化関数
	 */
	public Convolutionlayer(int[] imgsize, int chnl, int nkernel, int[] kernelsize, int[] convoutsize,
			int stride, int minibatch, Sfmt m, String actfunc) {
		// TODO 自動生成されたコンストラクター・スタブ
		if(m == null){
			int[] init_key = {(int) System.currentTimeMillis(), (int) Runtime.getRuntime().freeMemory()};
			m = new Sfmt(init_key);
		}
		mt = m;

		if(bias == null){
			//			bias = new float[nkernel];
			bias = new FloatMatrix(new float[nkernel]);
		}

		kernelnum = nkernel;
		chanel = chnl;
		this.kernelsize = kernelsize;
		imagesize = imgsize;
		this.convoutsize = convoutsize;
		padding=0;
		input_data = new FloatMatrix[minibatch][chnl];
		preact_data = new FloatMatrix[minibatch][nkernel];

//		for(int i=0; i<minibatch; i++){
//
//			for(int j=0; j<chnl; j++){
//				input_data[i][j] = new FloatMatrix(imgsize[0],imgsize[1]);
//			}
//			for(int j=0; j<nkernel; j++){
//				preact_data[i][j] = new FloatMatrix(convoutsize[0],convoutsize[1]);
//			}
//		}


		if(stride < 0){
			this.stride = 0;
		}else{
			this.stride = stride-1;
		}

		//		INDArray columnVectorA = Nd4j.create(new float[][], new int[]{nkernel, chnl});
		//INDArray columnVectorA = Nd4j.create(new float[nkernel*chnl*kernelsize[0]*kernelsize[1]], new int[]{nkernel, chnl, kernelsize[0], kernelsize[1]});
		//		INDArray matrixC = Nd4j.create(new double[8], new int[]{3, 3,3,3});
		//		INDArray matrixc = Nd4j.create(new double[][][][]);

//		FloatMatrix gg = new FloatMatrix(new float[3][4]);
		if(weight == null){
			weight = new FloatMatrix[nkernel][chnl];
			//			weight = new float[nkernel][chnl][kernelsize[0]][kernelsize[1]];
			adagrad_gradient = new float[nkernel][chnl][kernelsize[0]][kernelsize[1]];
			float in  = (float)(chnl * kernelsize[0] * kernelsize[1]);
			float out = (float)(nkernel * kernelsize[0] * kernelsize[1] / (convoutsize[0] * convoutsize[1]));
			float w = (float) Math.sqrt(6./(in+out));

			for(int kernel=0; kernel<nkernel; kernel++)
				for(int c=0; c<chnl; c++){
					weight[kernel][c] =  new FloatMatrix(kernelsize[0], kernelsize[1]);
					for(int ksize0=0; ksize0<kernelsize[0]; ksize0++)
						for(int ksize1=0; ksize1<kernelsize[1]; ksize1++){
							weight[kernel][c].put(ksize0,ksize1,RandomGenerator.uniform(-w, w, m));

							adagrad_gradient[kernel][c][ksize0][ksize1] = 0.f;
						}
				}
		}

		System.out.print("conv constractor " + convoutsize[0] +":"+ convoutsize[1]);
		System.out.println(". input channel:"+chnl+" output channel:" +nkernel);

		if(actfunc.equals("sigmoid")){
			System.out.println("activation is sigmoid");
			activation = (FloatMatrix x)->ActivationFunction.logistic_sigmoid(x);
			dactivation = (FloatMatrix x)->ActivationFunction.dsigmoid(x);
		}else if(actfunc.equals("tanh")){
			System.out.println("activation is tanh");
			activation = (FloatMatrix x)->ActivationFunction.tanh(x);
			dactivation = (FloatMatrix x)->ActivationFunction.dtanh(x);
		}else if(actfunc.equals("ReLU")){
			System.out.println("activation is ReLU");
			activation = (FloatMatrix x)->ActivationFunction.ReLU(x);
			dactivation = (FloatMatrix x)->ActivationFunction.dReLU(x);
		}else if(StringUtils.isEmpty(actfunc)){
			throw new IllegalArgumentException("specify activation function");
		}else{
			throw new IllegalArgumentException("activation function not supported");
		}
	}

	public FloatMatrix[] forward(FloatMatrix[] z, FloatMatrix[] preactdata, FloatMatrix[] after_act) {
		//		public float[][][] forward(float[][][] z, float[][][] preactdata, float[][][] after_act) {
		// TODO 自動生成されたメソッド・スタブ
		return convolve(z, preactdata, after_act);
	}

	public FloatMatrix[] convolve(FloatMatrix[] z, FloatMatrix[] preactdata, FloatMatrix[] after_act){
		//		public float[][][] convolve(float[][][] z, float[][][] preactdata, float[][][] after_act){
		Scanner scan = new Scanner(System.in);
		if(padding > 0){
			for(int i=0; i<z.length; i++){
				z[i] = Jblas_util.zeropadding(z[i], padding).dup();
			}
		}
		FloatMatrix[] y = new FloatMatrix[kernelnum];
		//		float[][][] y = new float[kernelnum][convoutsize[0]][convoutsize[1]];
		/*
		 * ストライドを加えるなら、Zのi,j部分にストライドを加える
		 * 配列オーバー部分には0を加える
		 */
		for(int kernel=0; kernel<kernelnum; kernel++){
//			System.out.println(after_act[kernel].rows+":"+after_act[kernel].columns+"--");
			y[kernel] = new FloatMatrix(new float[convoutsize[0]][convoutsize[1]]);
			FloatMatrix convol = new FloatMatrix(new float[convoutsize[0]][convoutsize[1]]);
			//怪しい.convoutではなくimgサイズ？zの長さではないか？
			for(int i=0; i<convoutsize[0]; i += 1+stride){
				//				for(int i=0; i<convoutsize[0]; i++)
				//カーネルが画像サイズを超えたらブレーク
				if(i+kernelsize[0] > z[0].getRows()){
					break;
				}

				for(int j=0; j<convoutsize[1]; j += 1+stride){
					//					for(int j=0; j<convoutsize[1]; j++){
					//					float convol = 0;
//					System.out.println((i+kernelsize[0])+"*"+(j+kernelsize[0])+"=" +z[0].getRows()+"*"+z[0].getColumns());
					if(j+kernelsize[0] > z[0].getColumns()){
						break;
					}
					for(int c=0; c<chanel; c++){
						//入力値をカーネルサイズ分切り取り
						FloatMatrix getkernel = z[c].getRange(i, i+kernelsize[0], j, j+kernelsize[1]);
						//畳込み結果を該当座標にぷっと
//						System.out.println(i+":"+j+":"+kernel+":"+c);
						if(i == 0 && j==0){
							convol.put(i, j, convol.get(i,j)+weight[kernel][c].mul(getkernel).sum());
						}else if( i==0 && j!=0){
							convol.put(i, j-stride,convol.get(i,j-stride)+weight[kernel][c].mul(getkernel).sum());
						}else if( i!=0&&j==0){
							convol.put(i-stride, j, convol.get(i-stride,j)+weight[kernel][c].mul(getkernel).sum());
						}else{
							convol.put(i-stride, j-stride, convol.get(i-stride,j-stride)+weight[kernel][c].mul(getkernel).sum());
						}
						//						for(int ks0=0; ks0<kernelsize[0]; ks0++)
						//							for(int ks1=0; ks1<kernelsize[1]; ks1++){
						//								System.out.println(kernel+":"+(i+ks0)+":"+(j+ks1));
						//								convol += weight[kernel][c][ks0][ks1] * z[c][i+ks0][j+ks1];
						//							}
					}
				}

				//このキャッシュはメインループ持たせるのではなく各層に持たせるべきでは？
				//System.out.println(kernel+":"+i+":"+j);
				//バイアスを加え活性化前後でキャッシュ

//				System.out.println(after_act[kernel]+"\n------");
//				System.out.println(preactdata[kernel]+"\n######");
//				System.out.println(activation.apply(convol));

//				System.out.println(after_act[kernel]+"\n++++++++++++++");
				//					preactdata[kernel][i][j] = convol + bias[kernel];
				//					after_act[kernel][i][j] = activation.apply(preactdata[kernel][i][j]);
				//					y[kernel][i][j] = after_act[kernel][i][j];
			}
			preactdata[kernel].addi(convol.addi(bias.get(kernel)));
			after_act[kernel].addi(activation.apply(convol));
			y[kernel] = activation.apply(convol);
//			scan.nextLine();
//			System.out.println(after_act[kernel].rows+":"+after_act[kernel].columns+":");
//			System.out.println(":"+after_act[kernel]);
		}
		scan.close();
		return y;
	}


	public FloatMatrix[] forward(FloatMatrix[] z,int n) {
		//		public float[][][] forward(float[][][] z, float[][][] preactdata, float[][][] after_act) {
		// TODO 自動生成されたメソッド・スタブ
		return convolve(z, n);
	}
/**
 * 畳み込みを行う
 * @param z 入力値
 * @param n ミニバッチ番号
 * @return 畳込み結果
 */
	public FloatMatrix[] convolve(FloatMatrix[] z, int n){

		if(padding > 0){
			for(int i=0; i<z.length; i++){
				z[i] = Jblas_util.zeropadding(z[i], padding).dup();
			}
		}

		FloatMatrix[] y = new FloatMatrix[kernelnum];
		/*
		 * ストライドを加えるなら、Zのi,j部分にストライドを加える
		 * 配列オーバー部分には0を加える
		 */
		//nが0未満なら判別時のForward
		if(n >=0){
			input_data[n] = z.clone();
		}
		for(int kernel=0; kernel<kernelnum; kernel++){
			//怪しい.convoutではなくimgサイズ？zの長さではないか？
			y[kernel] = new FloatMatrix(new float[convoutsize[0]][convoutsize[1]]);
			FloatMatrix convol = new FloatMatrix(new float[convoutsize[0]][convoutsize[1]]);
			for(int i=0; i<convoutsize[0]; i += 1+stride){
				//カーネルが画像サイズを超えたらブレーク
				if(i+kernelsize[0] > z[0].getRows()){
					break;
				}

				for(int j=0; j<convoutsize[1]; j += 1+stride){
					if(j+kernelsize[0] > z[0].getColumns()){
						break;
					}
					for(int c=0; c<chanel; c++){
						//入力値をカーネルサイズ分切り取り
						FloatMatrix getkernel = z[c].getRange(i, i+kernelsize[0], j, j+kernelsize[1]);
						//畳込み結果を該当座標にぷっと
						if(i == 0 && j==0){
							convol.put(i, j, convol.get(i,j)+weight[kernel][c].mul(getkernel).sum());
						}else if( i==0 && j!=0){
							convol.put(i, j-stride,convol.get(i,j-stride)+weight[kernel][c].mul(getkernel).sum());
						}else if( i!=0&&j==0){
							convol.put(i-stride, j, convol.get(i-stride,j)+weight[kernel][c].mul(getkernel).sum());
						}else{
							convol.put(i-stride, j-stride, convol.get(i-stride,j-stride)+weight[kernel][c].mul(getkernel).sum());
						}
					}
				}

				//このキャッシュはメインループ持たせるのではなく各層に持たせるべきでは？
				//バイアスを加え活性化前後でキャッシュ
				if(n>=0){
					preact_data[n][kernel] = convol.addi(bias.get(kernel)).dup();
				}
				y[kernel] = activation.apply(convol).dup();
			}
		}
		return y;
	}

	/**
	 *畳込み層の逆伝播を行う
	 * @param x 入力値
	 * @param preact 活性化する前の出力
	 * @param dy 逆伝播の値
	 * @param minibatchsize
	 * @param l_rate
	 * @return
	 */
	public FloatMatrix[][] backward(FloatMatrix[][] x , FloatMatrix[][] preact, FloatMatrix[][] dy,
			//			public float[][][][] backward(float[][][][] x , float[][][][] preact, float[][][][] dy,
			int minibatchsize, float l_rate) {
		// TODO 自動生成されたメソッド・スタブ
		return deconvolve(x, preact, dy, minibatchsize, l_rate);
	}

/**
 * 畳み込み層の逆伝播
 * @param x 層への入力
 * @param preact 活性化前
 * @param dy 前の層からの逆伝播
 * @param minibatchsize ミニバッチ数
 * @param l_rate 学習率
 * @return 逆伝播する誤差
 */
	private FloatMatrix[][] deconvolve(FloatMatrix[][] x, FloatMatrix[][] preact, FloatMatrix[][] dy,
			//			private float[][][][] deconvolve(float[][][][] x, float[][][][] preact, float[][][][] dy,
			int minibatchsize, float l_rate){
		FloatMatrix[][] grad_weight = new FloatMatrix[kernelnum][chanel];
		//		float[][][][] grad_weight = new float[kernelnum][chanel][kernelsize[0]][kernelsize[1]];
		//		float[] grad_bias = new float[kernelnum];
		FloatMatrix grad_bias = new FloatMatrix(new float[kernelnum]);

//		System.out.println(dy[0][0].rows+"*"+dy[0][0].columns+" = "+convoutsize[0]+":"+convoutsize[1]);

		for(int i=0; i<kernelnum; i++)
			for(int j=0; j<chanel; j++){
				grad_weight[i][j] = new FloatMatrix(new float[kernelsize[0]][kernelsize[1]]);
			}
		//重みとバイアスの勾配を計算.ここでアップデータの分岐を行うか
		//分岐をクラス生成時に選択するようにする
		for(int n=0; n<minibatchsize; n++){
			for(int k=0; k<kernelnum; k++){
				FloatMatrix d_ = dy[n][k].mul(dactivation.apply(preact[n][k])).dup();
				grad_bias.put(k, grad_bias.get(k)+d_.sum());
				for(int i=0; i<convoutsize[0]; i += 1+stride)
					for(int j=0; j<convoutsize[1]; j += 1+stride){
						for(int c=0; c<chanel; c++){
//							for(int s=0; i<kernelsize[0]; s++)
//								for(int t=0; j<kernelsize[1]; t++){
									FloatMatrix inputdata = x[n][c].getRange(i, i+kernelsize[0], j, j+kernelsize[1]);
									//							grad_weight[k][c].addi(d_.mul(x[n][c]));
//									System.out.println(d_.rows+":"+d_.columns+" = "+convoutsize[0]+":"+convoutsize[1]);
//									System.out.println(dy[n][k].rows+"*"+dy[n][k].columns+" = "+i+"*"+j);
									grad_weight[k][c].addi(inputdata.mul(d_.get(i, j)));
//								}
						}
					}

				//				for(int i=0; i<convoutsize[0]; i += 1+stride)
				//					for(int j=0; j<convoutsize[1]; j += 1+stride){
				//						//活性化の微分
				//						System.out.println(n+":"+k+":"+i+":"+j);
				//						float d_ = dy[n][k][i][j] * dactivation.apply(preact[n][k][i][j]);
				//						grad_bias[k] += d_; //バイアスの勾配
				//						//重みの勾配
				//						for(int c=0; c<chanel; c++)
				//							for(int s=0; s<kernelsize[0]; s++)
				//								for(int t=0; t<kernelsize[1]; t++){
				//									grad_weight[k][c][s][t] += d_ * x[n][c][i+s][j+t];
				//								}
				//					}
			}
		}

		//パラメータの更新
		bias.subi(grad_bias.mul(l_rate).div(minibatchsize));
		for(int k=0; k<kernelnum; k++){
			//			bias[k] -= l_rate * grad_bias[k] / minibatchsize;

			for(int c=0; c<chanel; c++){
				weight[k][c].subi(grad_weight[k][c].mul(l_rate).div(minibatchsize));
				//				for(int s=0; s<kernelsize[0]; s++)
				//					for(int t=0; t<kernelsize[1]; t++){
				//						weight[k][c][s][t] -= l_rate * grad_weight[k][c][s][t] / minibatchsize;
				//					}
			}
		}

		FloatMatrix[][] dx = new FloatMatrix[minibatchsize][chanel];
		//		float [][][][] dx = new float[minibatchsize][chanel][imagesize[0]][imagesize[1]];
		//逆伝播する誤差を計算
		for(int n=0; n<minibatchsize; n++)
			for(int c=0; c<chanel; c++){
//				for(int i=0; i<imagesize[0]; i++)
//					for(int j=0; j<imagesize[1]; j++){
						dx[n][c] = new FloatMatrix(new float[imagesize[0]][imagesize[1]]);
						//imagesizeは入力値
						for(int k=0; k<kernelnum; k++){
							FloatMatrix d_ = new FloatMatrix(new float[imagesize[0]][imagesize[1]]);
							FloatMatrix dy_ = dy[n][k].mul(dactivation.apply(preact[n][k]));
							FloatMatrix dd = new FloatMatrix(dy_.rows,dy_.columns);
							for(int a = 0; a<dy_.rows; a+=1+stride){
								for(int b=0; b<dy_.columns;b+=1+stride){
									if(a+kernelsize[0] > dy_.rows || b+kernelsize[1] > dy_.columns){
										break;
									}
									FloatMatrix getkernel = dy_.getRange(a, a+kernelsize[0], b, b+kernelsize[1]);
									Jblas_util.put(dd, getkernel.mul(weight[k][c]).add(dd.getRange(a, a+kernelsize[0], b, b+kernelsize[1])), a, b);
									//d_.put(a+dx_row, b+dx_column, getkernel.mul(weight[k][c]).sum());
								}
							}
							if(imagesize[0] - dy_.rows == 0 && imagesize[1] -dy_.columns == 0){
								d_ = dd.dup();
							}else if(imagesize[0] - dd.rows == 0){
								Jblas_util.put(d_, dd, 0, 1);
							}else if(imagesize[1] -dd.columns == 0){
								Jblas_util.put(d_, dd, 1, 0);
							}else if(imagesize[0] - dd.rows == 1 && imagesize[1] -dy_.columns == 1){
								Jblas_util.put(d_, dd, 1, 1);
							}else if(imagesize[0] - dd.rows == 1){
								Jblas_util.put(d_, dd, 1, Math.round((imagesize[1] - dy_.columns) / 2));
							}else if(imagesize[1] -dd.columns == 1){
								Jblas_util.put(d_, dd, Math.round((imagesize[0] -dy_.rows)/2), 1);
							}else{
								Jblas_util.put(d_, dd, Math.round((imagesize[0] -dy_.rows)/2),  Math.round((imagesize[1] - dy_.columns) / 2));
							}

							/*for(int s=0; s<kernelsize[0]; s++)
								for(int t=0; t<kernelsize[1]; t++){
									float d_ = 0;

									if(i-(kernelsize[0]-1)-s>=0 && j-(kernelsize[1]-1)-t>=0){
										d_ = dy[n][k][i-(kernelsize[0]-1)-s][j-(kernelsize[1]-1)-t]
												* dactivation.apply(preact[n][k][i-(kernelsize[0]-1)-s][j-(kernelsize[1]-1)-t])
												* weight[k][c][s][t];
									}
									dx[n][c][i][j] += d_;
								}*/

							dx[n][c].addi(d_);
						}
//					}
			}

		return dx;
	}

	/**
	 * 畳み込みの逆伝播
	 * @param dy 逆伝播してきた誤差
	 * @param minibatchsize ミニバッチ
	 * @param l_rate 学習率
	 * @return 逆伝播する誤差
	 */
	public FloatMatrix[][] backward(FloatMatrix[][] dy, int minibatchsize, float l_rate) {
		// TODO 自動生成されたメソッド・スタブ
		return deconvolve(dy, minibatchsize, l_rate);
	}

	/**
	 * 畳み込みの逆伝播の実処理
	 * @param dy 逆伝播してきた誤差
	 * @param minibatchsize ミニバッチ
	 * @param l_rate 学習率
	 * @return 逆伝播する誤差
	 */
	private FloatMatrix[][] deconvolve(FloatMatrix[][] dy, int minibatchsize, float l_rate){
		FloatMatrix[][] grad_weight = new FloatMatrix[kernelnum][chanel];
		FloatMatrix grad_bias = new FloatMatrix(new float[kernelnum]);

		for(int i=0; i<kernelnum; i++)
			for(int j=0; j<chanel; j++){
				grad_weight[i][j] = new FloatMatrix(new float[kernelsize[0]][kernelsize[1]]);
			}
		//重みとバイアスの勾配を計算.ここでアップデータの分岐を行うか
		//分岐をクラス生成時に選択するようにする
		for(int n=0; n<minibatchsize; n++){
			for(int k=0; k<kernelnum; k++){
				FloatMatrix d_ = dy[n][k].mul(dactivation.apply(preact_data[n][k])).dup();
				grad_bias.put(k, grad_bias.get(k)+d_.sum());
				for(int i=0; i<convoutsize[0]; i += 1+stride)
					for(int j=0; j<convoutsize[1]; j += 1+stride){
						for(int c=0; c<chanel; c++){
									FloatMatrix inputdata = input_data[n][c].getRange(i, i+kernelsize[0], j, j+kernelsize[1]);
									grad_weight[k][c].addi(inputdata.mul(d_.get(i, j)));
						}
					}
			}
		}

		//パラメータの更新
		bias.subi(grad_bias.mul(l_rate).div(minibatchsize));
		for(int k=0; k<kernelnum; k++){
			for(int c=0; c<chanel; c++){
				weight[k][c].subi(grad_weight[k][c].mul(l_rate).div(minibatchsize));
			}
		}

		FloatMatrix[][] dx = new FloatMatrix[minibatchsize][chanel];
		//逆伝播する誤差を計算
		for(int n=0; n<minibatchsize; n++)
			for(int c=0; c<chanel; c++){
						dx[n][c] = new FloatMatrix(new float[imagesize[0]][imagesize[1]]);
						//imagesizeは入力値
						for(int k=0; k<kernelnum; k++){
							FloatMatrix d_ = new FloatMatrix(new float[imagesize[0]][imagesize[1]]);
							FloatMatrix dy_ = dy[n][k].mul(dactivation.apply(preact_data[n][k]));
							FloatMatrix dd = new FloatMatrix(dy_.rows,dy_.columns);
							for(int a = 0; a<dy_.rows; a+=1+stride){
								for(int b=0; b<dy_.columns;b+=1+stride){
									if(a+kernelsize[0] > dy_.rows || b+kernelsize[1] > dy_.columns){
										break;
									}
									FloatMatrix getkernel = dy_.getRange(a, a+kernelsize[0], b, b+kernelsize[1]);
									Jblas_util.put(dd, getkernel.mul(weight[k][c]).add(dd.getRange(a, a+kernelsize[0], b, b+kernelsize[1])), a, b);
								}
							}
							if(imagesize[0] - dy_.rows == 0 && imagesize[1] -dy_.columns == 0){
								d_ = dd.dup();
							}else if(imagesize[0] - dd.rows == 0){
								Jblas_util.put(d_, dd, 0, 1);
							}else if(imagesize[1] -dd.columns == 0){
								Jblas_util.put(d_, dd, 1, 0);
							}else if(imagesize[0] - dd.rows == 1 && imagesize[1] -dy_.columns == 1){
								Jblas_util.put(d_, dd, 1, 1);
							}else if(imagesize[0] - dd.rows == 1){
								Jblas_util.put(d_, dd, 1, Math.round((imagesize[1] - dy_.columns) / 2));
							}else if(imagesize[1] -dd.columns == 1){
								Jblas_util.put(d_, dd, Math.round((imagesize[0] -dy_.rows)/2), 1);
							}else{
								Jblas_util.put(d_, dd, Math.round((imagesize[0] -dy_.rows)/2),  Math.round((imagesize[1] - dy_.columns) / 2));
							}
							dx[n][c].addi(d_);
						}
			}

		return dx;
	}
}
