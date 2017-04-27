package layers;

public class LRNlayer {
	private float alpha;
	private float beta;
	private float k;
	private int local_size;
	float[][][][] input, val, scale;
//	private int chanel;

	public LRNlayer(float a, float b, float k, int local){
		//, int chanel){
		if(a > 0){
			alpha = a;
		}else{
			alpha = (float)1e-4;
		}

		if(b > 0){
			beta=b;
		}else{
			beta=0.75f;
		}

		if(local > 0){
			local_size = local;
		}else{
			local_size = 5;
		}

		if(k > 0){
			this.k = k;
		}else{
			k = 2;
		}

//		this.chanel = chanel;
	}


	public float[][][][] forward(float[][][][] z, int minibatch, int chanel){

		int size = z[0][0].length, size1 = z[0][0][0].length;
		float tmp=0.f;

		float[][][][] y = z.clone();
		input = z.clone();
		val = scale = z.clone();
		for(int m=0; m < minibatch; m++){

			for(int i=0; i< size; i++){
				for(int j=0; j<size1; j++){
					for(int c=0; c<chanel; c++){
						int c_ = c - local_size/2;
						if(c_ < 0){
							for(c_ = 0; c_ < local_size/2; c_++){
								tmp += z[m][c_][i][j] * z[m][c_][i][j];
							}
						}else if (0 <= c_ && c+local_size < chanel ){
							for(; c_ < c+local_size; c_++){
								tmp += z[m][c_][i][j] * z[m][c_][i][j];
							}
						}
						scale[m][c][i][j] = k+alpha*tmp;
						val[m][c][i][j] = (float) Math.pow(scale[m][c][i][j], beta);
						y[m][c][i][j] = z[m][c][i][j] / val[m][c][i][j];
					}

				}


			}

		}

		return y;
	}


	public float[][][][] backward(float[][][][] dy){
		int minibatch = input.length, chanel = input[0].length;
		int size = input[0][0].length, size1 = input[0][0][0].length;
		float[][][][] tmp = input.clone();
		float[][] sum_error = new float[size][size1];

		for(int m=0; m<minibatch; m++)
			for(int c=0; c< chanel; c++)
				for(int i=0; i<size; i++)
					for(int j=0; j<size1; j++){
						tmp[m][c][i][j] *= dy[m][c][i][j];
						sum_error[i][j] += tmp[m][c][i][j];
					}


		// gx = gy * val - 2 * alpha * beta * sumPart/unitScale * a^i_{x,y}
		float[][][][] error = dy.clone();

		for(int m=0; m<minibatch; m++)
			for(int c=0; c< chanel; c++)
				for(int i=0; i<size; i++)
					for(int j=0; j<size1; j++){
						error[m][c][i][j] = dy[m][c][i][j] * val[m][c][i][j]
								- (2 * alpha * beta * sum_error[i][j] / (scale[m][c][i][j] * input[m][c][i][j]));
					}

		return error;
	}



	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ
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

		LRNlayer b = new LRNlayer(0,0,0,0);

		b.forward(z, minibatch, chanel);
		b.backward(dy);
		System.out.println("end");
	}

}
