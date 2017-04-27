package layers;

import org.apache.commons.lang3.StringUtils;
import org.jblas.FloatMatrix;

import util.ActivationFunction;
import util.Jblas_util;
import Mersenne.Sfmt;

public class AvePoolinglayer {
	int[] poolkernelsize;
	int[] pooloutsize;
	Sfmt mt;
	int flatsize;
	int kernelnum;
	String type;
	ActivationFunction.FloatMatrixFunction<FloatMatrix> activation;
	ActivationFunction.FloatMatrixFunction<FloatMatrix> dactivation;
	//	FloatFunction<Float> activation;
//	FloatFunction<Float> dactivation;
	int stride;
	int avesize;
	int padding=0;

	FloatMatrix[][] input_data;
	FloatMatrix[][] output_data;

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
	 * プーリング層のコンストラクタ
	 * @param pkernelsize プーリング層のカーネルサイズ
	 * @param poutsize プーリング層の出力数
	 * @param m メルセンヌツイスタ
	 */
	public AvePoolinglayer(int[] pkernelsize, int[] poutsize, int nkernel, int stride,int minibatch, Sfmt m, String poolingtype, String actfunc){
		poolkernelsize = pkernelsize;
		pooloutsize = poutsize;
		kernelnum = nkernel;
		mt = m;
		avesize = pkernelsize[0]*pkernelsize[1];
		input_data = new FloatMatrix[minibatch][nkernel];
		output_data = new FloatMatrix[minibatch][nkernel];

		if(stride <= 0){
			this.stride = 0;
		}else{
			this.stride = stride-1;
		}

		if(actfunc.equals("sigmoid")){
			activation = (FloatMatrix x)->ActivationFunction.logistic_sigmoid(x);
			dactivation = (FloatMatrix x)->ActivationFunction.dsigmoid(x);
		}else if(actfunc.equals("tanh")){
			activation = (FloatMatrix x)->ActivationFunction.tanh(x);
			dactivation = (FloatMatrix x)->ActivationFunction.dtanh(x);
		}else if(actfunc.equals("ReLU")){
			activation = (FloatMatrix x)->ActivationFunction.ReLU(x);
			dactivation = (FloatMatrix x)->ActivationFunction.dReLU(x);
		}else{
			activation = null;
			dactivation = null;
		}

		if(poolingtype.equals("MAX") || StringUtils.isEmpty(poolingtype)){
			System.out.println("MAX");
			type = "MAX";
		}else if(poolingtype.equals("AVE")){
			type = "AVE";
		}else{
			System.out.println("error");
			throw new IllegalArgumentException("specify poolingtype function");
		}

	}

	/**
	 * マックスプーリングで順伝播
	 * @param z 入力値
	 * @return 順伝播の値
	 */
	public FloatMatrix[] avepooling(FloatMatrix[] z) {
//		public float[][][] avepooling(float[][][] z) {
		// TODO 自動生成されたメソッド・スタブ
		FloatMatrix[] y = new FloatMatrix[kernelnum];
//		float[][][] y = new float[kernelnum][pooloutsize[0]][pooloutsize[1]];

		for(int kernel=0; kernel<kernelnum; kernel++)
			for(int i=0; i<pooloutsize[0]; i+=1+stride)
				for(int j=0; j<pooloutsize[1]; j+=1+stride){
//					for(int i=0; i<pooloutsize[0]; i++)
//						for(int j=0; j<pooloutsize[1]; j++){
//					float ave = 0.f;
//					FloatMatrix getkernel = z[kernel].getRange(i, i+poolkernelsize[0], j, j+poolkernelsize[1]);

					y[kernel].put(i, j, z[kernel].getRange(i, i+poolkernelsize[0], j, j+poolkernelsize[1]).mean());
//					for(int ks0=0; ks0<poolkernelsize[0]; ks0++)
//						for(int ks1=0; ks1<poolkernelsize[1]; ks1++){
//								ave += z[kernel][poolkernelsize[0]*i+ks0][poolkernelsize[1]*j+ks1];
//						}

//					y[kernel][i][j] = ave / avesize;
				}
		return y;
	}

	/**
	 * マックスプーリングで順伝播
	 * @param z 入力値
	 * @return 順伝播の値
	 */
	public FloatMatrix[] avepooling(FloatMatrix[] z, int n) {
		// TODO 自動生成されたメソッド・スタブ
		FloatMatrix[] y = new FloatMatrix[kernelnum];

		for(int kernel=0; kernel<kernelnum; kernel++){
			y[kernel] = new FloatMatrix(pooloutsize[0], pooloutsize[1]);
			for(int i=0; i<pooloutsize[0]; i+=1+stride){
				for(int j=0; j<pooloutsize[1]; j+=1+stride){
					y[kernel].put(i, j, z[kernel].getRange(i, i+poolkernelsize[0], j, j+poolkernelsize[1]).mean());
				}
			}
		}

		if(n >= 0){
			input_data[n] = z.clone();
			output_data[n] = y.clone();
		}
		return y;
	}

	/**
	 * アベレージプーリングの逆伝播
	 * @param x 入力
	 * @param y 出力
	 * @param dy 逆伝播の値
	 * @param convoutsize 畳込み層のアウトプット数
	 * @param minibatchsize ミニバッチサイズ
	 * @return 逆伝播の値
	 */
	public FloatMatrix[][] backavepooing(FloatMatrix[][] x, FloatMatrix[][]y, FloatMatrix[][] dy,int[] convoutsize, int minibatchsize){
//		public float[][][][] backmaxpooing(float[][][][] x, float[][][][]y, float[][][][] dy,int[] convoutsize, int minibatchsize){
		FloatMatrix[][] back = new FloatMatrix[minibatchsize][kernelnum];
//		float[][][][] back = new float[minibatchsize][kernelnum][convoutsize[0]][convoutsize[1]];

		for(int n=0; n<minibatchsize; n++){
			for(int kernel=0; kernel<kernelnum; kernel++){
				back[n][kernel] = new FloatMatrix(convoutsize[0], convoutsize[1]);
				for(int i=0; i<pooloutsize[0]; i+=1+stride){
					for(int j=0; j<pooloutsize[1]; j+=1+stride){
						FloatMatrix getkernel = new FloatMatrix(poolkernelsize[0], poolkernelsize[1]).add((float) 1/poolkernelsize[0]*poolkernelsize[1]);
//						getkernel.eqi(y[n][kernel].get(i,j));
						getkernel.mul(dy[n][kernel].get(i,j));

						//for(int s=0; s<getkernel.rows;s++){
							for(int t=0; t<getkernel.columns; t++){
								back[n][kernel].putColumn(poolkernelsize[1]*j+t, getkernel);
//								back[n][kernel].put(poolkernelsize[0]*i+s, poolkernelsize[1]*j+t, getkernel.get(i,j));
							}
						//}
//						for(int s=0; s<poolkernelsize[0]; s++)
//							for(int t=0; t<poolkernelsize[1]; t++){
//								float d = 0.f;
//								//System.out.println(n+":"+kernel+":"+(poolkernelsize[0]*i+s)+":"+(poolkernelsize[1]*j+t));
//								if(y[n][kernel][i][j] == x[n][kernel][poolkernelsize[0]*i+s][poolkernelsize[1]*j+t]){
//									d = dy[n][kernel][i][j];
//								}
//								back[n][kernel][poolkernelsize[0]*i+s][poolkernelsize[1]*j+t] = d;
//							}

					}
				}
			}
		}

		return back;
	}


	/**
	 * アベレージプーリングの逆伝播
	 * @param dy 逆伝播の値
	 * @param convoutsize 畳込み層のアウトプット数
	 * @param minibatchsize ミニバッチサイズ
	 * @return 逆伝播の値
	 */
	public FloatMatrix[][] backavepooing(FloatMatrix[][] dy,int[] convoutsize, int minibatchsize){
		FloatMatrix[][] back = new FloatMatrix[minibatchsize][kernelnum];
		for(int n=0; n<minibatchsize; n++){
			for(int kernel=0; kernel<kernelnum; kernel++){
				back[n][kernel] = new FloatMatrix(input_data[n][kernel].rows,
						input_data[n][kernel].columns);//.add(1.f/(poolkernelsize[0]*poolkernelsize[1]));

				for(int i=0; i<pooloutsize[0]; i+=1+stride){
					if(i+poolkernelsize[0] >= back[n][kernel].rows){
						break;
					}
					for(int j=0; j<pooloutsize[1]; j+=1+stride){
						if(j+poolkernelsize[1] >= back[n][kernel].getColumns()){
							break;
						}
						FloatMatrix getkernel = back[n][kernel].getRange(i, i+poolkernelsize[0], j, j+poolkernelsize[1]);//.add(output_data[n][kernel].get(i,j));

						if(activation == null){
							getkernel.addi(1.f/(poolkernelsize[0]*poolkernelsize[1])).muli(dy[n][kernel].get(i,j)).muli(output_data[n][kernel].get(i,j));
						}else{
							getkernel.addi(1.f/(poolkernelsize[0]*poolkernelsize[1])).muli(dy[n][kernel].get(i,j))
							.muli(dactivation.apply(output_data[n][kernel]).get(i,j));
						}
						Jblas_util.put(back[n][kernel], getkernel, i, j);
					}
				}
			}
		}

		return back;
	}


}
