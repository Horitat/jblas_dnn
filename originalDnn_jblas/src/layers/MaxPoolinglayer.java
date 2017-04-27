package layers;

import java.util.Scanner;

import org.apache.commons.lang3.StringUtils;
import org.jblas.FloatMatrix;

import util.ActivationFunction;
import util.Jblas_util;
import Mersenne.Sfmt;

public class MaxPoolinglayer {

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
	int padding;

	FloatMatrix[][] input_data;
	FloatMatrix[][] output_data;

	/**
	 * プーリング層のコンストラクタ
	 * @param pkernelsize プーリング層のカーネルサイズ
	 * @param poutsize プーリング層の出力数
	 * @param nkernel カーネル数、チャネル
	 * @param stride スライド数
	 * @param m メルセンヌツイスタ
	 * @param poolingtype プーリングタイプ、不要か？
	 * @param actfunc 活性化関数
	 */
	public MaxPoolinglayer(int[] pkernelsize, int[] poutsize, int nkernel, int stride, int minibatch,Sfmt m, String poolingtype, String actfunc){
		poolkernelsize = pkernelsize;
		pooloutsize = poutsize;
		kernelnum = nkernel;
		mt = m;

		input_data = new FloatMatrix[minibatch][nkernel];
		output_data = new FloatMatrix[minibatch][nkernel];
		if(stride <= 0){
			this.stride = 0;
		}else{
			this.stride = stride-1;
		}
		padding=0;

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
		System.out.print("Create pool ");
		if(poolingtype.equals("MAX") || StringUtils.isEmpty(poolingtype)){
			System.out.print( "MAX");
			type = "MAX";
		}else if(poolingtype.equals("AVE")){
			System.out.print( "AVE");
			type = "AVE";
		}else{
			System.out.println("error");
			throw new IllegalArgumentException("specify poolingtype function");
		}

		System.out.println(". chnnel is " + nkernel);
	}

	/**
	 * マックスプーリングで順伝播
	 * @param z 入力値
	 * @return 順伝播の値
	 */
	public FloatMatrix[] maxpooling(FloatMatrix[] z) {
		//		public float[][][] maxpooling(float[][][] z) {
		// TODO 自動生成されたメソッド・スタブ
		FloatMatrix[] y = new FloatMatrix[kernelnum];
		//		float[][][] y = new float[kernelnum][pooloutsize[0]][pooloutsize[1]];

		if(padding > 0){
			for(int i=0; i<z.length; i++){
				z[i] = Jblas_util.zeropadding(z[i], padding).dup();
			}
		}

		for(int kernel=0; kernel<kernelnum; kernel++){
			FloatMatrix max = new FloatMatrix(pooloutsize[0], pooloutsize[1]);
			for(int i=0; i<pooloutsize[0]; i += 1+stride){
				if((i+poolkernelsize[0]) >= z[kernel].rows ){
					break;
				}
				for(int j=0; j<pooloutsize[1]; j += 1+stride){
					//					for(int i=0; i<pooloutsize[0]; i++)
					//						for(int j=0; j<pooloutsize[1]; j++){

					if((j+poolkernelsize[1]) >= z[kernel].columns){
//						System.out.println("break");
						break;
					}
//					System.out.println(i+"'&"+(i+poolkernelsize[0])+"^^"+(j+poolkernelsize[1])+"--"+z[kernel].rows+"::"+z[kernel].columns);
//					System.out.println(z[kernel]);
					FloatMatrix getkernel = z[kernel].getRange(i, i+poolkernelsize[0], j, j+poolkernelsize[1]);
//					System.out.println("kernel"+getkernel);
//					System.out.println(i+"#"+j+":"+getkernel.max());
					if(i == 0 && j==0){
						max.put(i, j, getkernel.max());
					}else if( i==0 && j!=0){
						max.put(i, j-stride, getkernel.max());
					}else if( i!=0&&j==0){
						max.put(i-stride, j, getkernel.max());
					}else{
						max.put(i-stride, j-stride, getkernel.max());
					}
//					System.out.println("max"+max);
					//					for(int ks0=0; ks0<poolkernelsize[0]; ks0++)
					//						for(int ks1=0; ks1<poolkernelsize[1]; ks1++){
					//							if(ks0==0 && ks1==0){
					//								max = z[kernel][poolkernelsize[0]*i][poolkernelsize[1]*j];
					//								continue;
					//							}
					//
					//							if(max < z[kernel][poolkernelsize[0]*i+ks0][poolkernelsize[1]*j+ks1]){
					//								max = z[kernel][poolkernelsize[0]*i+ks0][poolkernelsize[1]*j+ks1];
					//							}
					//						}
				}
			}
			if(activation != null){
				max = activation.apply(max);
			}
//			System.out.println( z[kernel]);
			y[kernel] = max.dup();
		}

		return y;
	}

	/**
	 * マックスプーリング
	 * @param z 入力
	 * @param n ミニバッチの番号
	 * @return プーリング結果
	 */
	public FloatMatrix[] maxpooling(FloatMatrix[] z, int n) {
		FloatMatrix[] y = new FloatMatrix[kernelnum];

		if(padding > 0){
			for(int i=0; i<z.length; i++){
				z[i] = Jblas_util.zeropadding(z[i], padding).dup();
			}
		}

		if(n>=0){
			input_data[n]= z.clone();
		}
		for(int kernel=0; kernel<kernelnum; kernel++){
			FloatMatrix max = new FloatMatrix(pooloutsize[0], pooloutsize[1]);
			for(int i=0; i<pooloutsize[0]; i += 1+stride){
				if((i+poolkernelsize[0]) >= z[kernel].rows ){
					break;
				}
				for(int j=0; j<pooloutsize[1]; j += 1+stride){

					if((j+poolkernelsize[1]) >= z[kernel].columns){
						break;
					}
					FloatMatrix getkernel = z[kernel].getRange(i, i+poolkernelsize[0], j, j+poolkernelsize[1]);
					if(i == 0 && j==0){
						max.put(i, j, getkernel.max());
					}else if( i==0 && j!=0){
						max.put(i, j-stride, getkernel.max());
					}else if( i!=0&&j==0){
						max.put(i-stride, j, getkernel.max());
					}else{
						max.put(i-stride, j-stride, getkernel.max());
					}
				}
			}
			if(activation != null){
				max = activation.apply(max);
			}
			if(n >=0){
				output_data[n][kernel] = max.dup();
			}
			y[kernel] = max.dup();
		}
		return y;
	}

	/**
	 * マックスプーリングの逆伝播
	 * @param x 入力
	 * @param y 出力
	 * @param dy 逆伝播の値
	 * @param convoutsize 畳込み層のアウトプット数
	 * @param minibatchsize ミニバッチサイズ
	 * @return 逆伝播の値
	 */
	public FloatMatrix[][] backmaxpooing(FloatMatrix[][] x, FloatMatrix[][] y, FloatMatrix[][] dy,int[] convoutsize, int minibatchsize){
		FloatMatrix[][] back = new FloatMatrix[minibatchsize][kernelnum];
		//		public float[][][][] backmaxpooing(float[][][][] x, float[][][][] y, float[][][][] dy,int[] convoutsize, int minibatchsize){
		//			float[][][][] back = new float[minibatchsize][kernelnum][convoutsize[0]][convoutsize[1]];

		Scanner sc = new Scanner(System.in);
		for(int n=0; n<minibatchsize; n++){
			for(int kernel=0; kernel<kernelnum; kernel++){
				back[n][kernel] = new FloatMatrix(x[n][kernel].rows,x[n][kernel].columns);
				FloatMatrix x_y_eq = Jblas_util.element_eq(x[n][kernel], y[n][kernel]);

				//				System.out.println(convoutsize[0]+"'"+convoutsize[1]+" = "+x[n][kernel].rows+"*"+x[n][kernel].columns);
//				System.out.println(pooloutsize[0]+"__"+pooloutsize[1]);
				for(int i=0; i<pooloutsize[0]; i+=1+stride){
					if(i+poolkernelsize[0] >= x[n][kernel].rows){
						break;
					}
					for(int j=0; j<pooloutsize[1]; j+=1+stride){

						if(j+poolkernelsize[1] >= x[n][kernel].getColumns()){
							break;
						}
//						System.out.println((i+poolkernelsize[0])+"__"+(j+poolkernelsize[1])+"--"+x[n][kernel].rows+"::"+x[n][kernel].columns);
//						System.out.println((i+poolkernelsize[0])+"__"+(j+poolkernelsize[1])+"--"+x_y_eq.rows+"::"+x_y_eq.columns);
//						System.out.println(dy[n][kernel].rows+"::"+dy[n][kernel].columns);
//						System.out.println(x_y_eq.rows+";;"+x_y_eq.columns);
						FloatMatrix getkernel = x_y_eq.getRange(i,poolkernelsize[0]+i, j,poolkernelsize[1]+j);
//						getkernel.eqi(y[n][kernel].get(i,j));

						/*
						 * 入力値からカーネルサイズを抜出し、一番大きい値を対応部分に入れ込む
						 */
//						System.out.println(x[n][kernel]+"\n))))))");
//						System.out.println(y[n][kernel]+"\n*/*/*/*/*");
//						System.out.println(x_y_eq+"\n+-+-+-)");
//						System.out.println("get kernel ");
//						System.out.println(getkernel+"\n//// "+dy[n][kernel].get(i,j)+" ////");
//						System.out.println(back[n][kernel]+"\n*******************");
						getkernel.muli(dy[n][kernel].get(i,j));
//						System.out.println(getkernel+"\n//// ////");
						Jblas_util.put(back[n][kernel], getkernel, i, j);
//						for(int s=0; s<getkernel.rows;s++){
//							for(int t=0; t<getkernel.columns; t++){
//								System.out.println((poolkernelsize[0]*i+s)+"$"+(poolkernelsize[1]*j+t)+"="+back[n][kernel].rows+"#"+back[n][kernel].columns);
//								back[n][kernel].put(poolkernelsize[0]*i+s, poolkernelsize[1]*j+t, getkernel.max());
//							}
//						}
//						System.out.println(back[n][kernel]+"\n**");

						//						for(int s=0; s<poolkernelsize[0]; s++){
						//							for(int t=0; t<poolkernelsize[1]; t++){
						//								float d = 0.f;
						//								//System.out.println(n+":"+kernel+":"+(poolkernelsize[0]*i+s)+":"+(poolkernelsize[1]*j+t));
						//								if(y[n][kernel][i][j] == x[n][kernel][poolkernelsize[0]*i+s][poolkernelsize[1]*j+t]){
						//									//活性化関数があれば活性化関数の微分を行う
						//									d = dy[n][kernel][i][j];
						//								}
						//								if(dactivation != null){
						//									d = dactivation.apply(d);
						//								}
						//								back[n][kernel] = d;
						//								back[n][kernel][poolkernelsize[0]*i+s][poolkernelsize[1]*j+t] = d;
						//							}
						//						}
					}
				}
//				sc.nextLine();
			}
		}
		sc.close();
		return back;
	}

	/**
	 * マックスプーリングの逆伝播
	 * @param dy 逆伝播してきた誤差
	 * @param convoutsize
	 * @param minibatchsize
	 * @return 逆伝播する誤差
	 */
	public FloatMatrix[][] backmaxpooing(FloatMatrix[][] dy,int[] convoutsize, int minibatchsize){
		FloatMatrix[][] back = new FloatMatrix[minibatchsize][kernelnum];

		for(int n=0; n<minibatchsize; n++){
			for(int kernel=0; kernel<kernelnum; kernel++){
				back[n][kernel] = new FloatMatrix(input_data[n][kernel].rows,input_data[n][kernel].columns);
				FloatMatrix x_y_eq = Jblas_util.element_eq(input_data[n][kernel], output_data[n][kernel]);

				for(int i=0; i<pooloutsize[0]; i+=1+stride){
					if(i+poolkernelsize[0] >= input_data[n][kernel].rows){
						break;
					}
					for(int j=0; j<pooloutsize[1]; j+=1+stride){

						if(j+poolkernelsize[1] >= input_data[n][kernel].getColumns()){
							break;
						}
						FloatMatrix getkernel = x_y_eq.getRange(i,poolkernelsize[0]+i, j,poolkernelsize[1]+j);
						/*
						 * 入力値からカーネルサイズを抜出し、一番大きい値を対応部分に入れ込む
						 */
						getkernel.muli(dy[n][kernel].get(i,j));
						Jblas_util.put(back[n][kernel], getkernel, i, j);
					}
				}
			}
		}

		return back;
	}


}
