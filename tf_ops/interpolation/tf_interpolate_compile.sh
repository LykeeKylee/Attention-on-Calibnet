g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /home/lykee/Junior/Apps/anaconda3/envs/tfenv1/lib/python3.6/site-packages/tensorflow/include  -I /usr/local/cuda/include -lcudart -L /usr/local/cuda/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0
echo '***********************'
