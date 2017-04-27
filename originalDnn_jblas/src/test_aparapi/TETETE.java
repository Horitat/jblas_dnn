package test_aparapi;


import com.amd.aparapi.Kernel;
import com.amd.aparapi.Range;




public class TETETE {

	@SuppressWarnings("deprecation")
	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ
        // GPUと授受する変数を定義(final変数のみ授受可能)
		final float a[]         = new float[]{ 1.0f , 2.0f , 3.0f , 4.0f };
		final float b[]         = new float[]{ 1.0f , 2.0f , 3.0f , 4.0f };
		final float result[]    = new float[ a.length ];

		// GPGPU用の無名クラスを生成
		Kernel kernel   = new Kernel()
		{
			@Override
			public void run()
			{
				// 配列の加算を実行
				int i       = getGlobalId();
				result[i]   = a[i] + b[i];
			}
		};

		// 配列の要素数を指定して、GPGPU実行
		Range range = Range.create( result.length );
		kernel.execute( range );

		// 実行結果を出力
		for( float el : result ){ System.out.println( el ); }

		// 実行モードを出力
		System.out.println( kernel.getExecutionMode().name() );
		System.out.println(kernel.getExecutionMode().isOpenCL() ? "OpenCL" : "Not OpenCL");
		// GPGPU用の無名クラスを解放
		kernel.dispose();
	}

}
