package layers;

import lombok.Data;

@Data
public class BatchNormlayer {

	private float gamma;
	private float beta;
	private float[][][][] xc,xn;
	private float std;
	final private float param = (float) 1e-8;;

	public BatchNormlayer(){
		gamma = 1;
		beta = 0;

	}

	/**
	 * バッチノーマライゼーション
	 * 畳込み層かプーリング層から
	 * @param z 入力値
	 * @param minibatch バッチ数
	 * @param chanel チャネル数(kernelnum)
	 * @return 結果
	 */
	public float[][][][] forward(float[][][][] z, int minibatch, int chanel){
		int size = z[0][0].length, size1 = z[0][0][0].length;
		float mu = 0;
		for(int mb_size =0; mb_size<minibatch; mb_size++)
			for(int c=0; c<chanel; c++)
				for(int i=0; i<size; i++)
					for(int j=0; j<size1; j++){
						mu += (float)(z[mb_size][c][i][j] / minibatch);
					}

		float delta = 0;
		xc = z.clone();
		for(int mb_size =0; mb_size<minibatch; mb_size++)
			for(int c=0; c<z[mb_size].length; c++)
				for(int i=0; i<size; i++)
					for(int j=0; j<size1; j++){
						xc[mb_size][c][i][j] = (float)((z[mb_size][c][i][j] - mu));
						delta += (float)(xc[mb_size][c][i][j]* xc[mb_size][c][i][j] / minibatch);
					}

		std = (float) Math.sqrt(delta+param);
		float[][][][] result = new  float[minibatch][chanel][size][size1];
		xn = xc.clone();
		for(int mb_size =0; mb_size<minibatch; mb_size++)
			for(int c=0; c<z[mb_size].length; c++)
				for(int i=0; i<size; i++)
					for(int j=0; j<size1; j++){
						xn[mb_size][c][i][j] = xc[mb_size][c][i][j] / std;
						result[mb_size][c][i][j] = gamma * xn[mb_size][c][i][j] + beta;
					}

		return result;
	}


	public float[][][][] backward(float[][][][] x , float[][][][] dy, int minibatch, float l_rate){
		float dbeta=0;
		for(int m=0; m<dy.length; m++)
			for(int c=0; c<dy[0].length; c++)
				for(int i=0; i<dy[0][0].length; i++)
					for(int j=0; j<dy[0][0][0].length; j++){
						dbeta += dy[m][c][i][j];
					}

		float dgamma=0, dstd=0;
		float[][][][] dxn = dy.clone(), dxc = dy.clone();
		for(int m=0; m<xn.length; m++)
			for(int c=0; c<xn[0].length; c++)
				for(int i=0; i<xn[0][0].length; i++)
					for(int j=0; j<xn[0][0][0].length; j++){
						dgamma += xn[m][c][i][j] * dy[m][c][i][j];
						dxn[m][c][i][j] = gamma * dy[m][c][i][j];
						dxc[m][c][i][j] = dxn[m][c][i][j] / std;
						dstd += (dxn[m][c][i][j] * xc[m][c][i][j]) / (std*std);
					}

		float ddelta = 0.5f * dstd/std;

		float dmu = 0;
		for(int m=0; m<xn.length; m++)
			for(int c=0; c<xn[0].length; c++)
				for(int i=0; i<xn[0][0].length; i++)
					for(int j=0; j<xn[0][0][0].length; j++){
						dxc[m][c][i][j] +=(2.f/minibatch) *dxc[m][c][i][j] * ddelta;
						dmu += dxc[m][c][i][j];
					}

		float[][][][] dx = dxc.clone();

		for(int m=0; m<xn.length; m++)
			for(int c=0; c<xn[0].length; c++)
				for(int i=0; i<xn[0][0].length; i++)
					for(int j=0; j<xn[0][0][0].length; j++){
						dx[m][c][i][j] = dxc[m][c][i][j] - dmu / minibatch;
					}

		return dx;
	}

	public static void main(String[] args) {

		int minibatch = 50;
		int chanel = 3;
		int size =10, size2 = 5;

		float[][][][] dy = new float[minibatch][chanel][size][size2];
		float[][][][] z = new float[minibatch][chanel][size][size2];
		int n=0;
		for(int mb_size =0; mb_size<minibatch; mb_size++){
			System.out.println("dy");
			for(int c=0; c<chanel; c++){
				System.out.print("dy"+c+"\n");
				for(int i=0; i<size; i++){
					for(int j=0; j<size2; j++){
						dy [mb_size][c][i][j]= n*j;
						z[mb_size][c][i][j] = n;
						n++;
						System.out.print(dy [mb_size][c][i][j]+",");
					}
					System.out.println();
				}
			}
		}

		for(int mb_size =0; mb_size<minibatch; mb_size++){
			System.out.println("z");
			for(int c=0; c<chanel; c++){
				System.out.print("z"+c+"\n");
				for(int i=0; i<size; i++){
					for(int j=0; j<size2; j++){
						System.out.print(z[mb_size][c][i][j]+",");
					}
					System.out.println();
				}
			}
		}

		BatchNormlayer b = new BatchNormlayer();

		b.forward(z, minibatch, chanel);
		b.backward(z, dy, minibatch, 0.001f);
		System.out.println("end");
	}
}
