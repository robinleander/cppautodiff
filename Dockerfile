FROM alpine:edge


WORKDIR /app
RUN apk update
RUN apk add git cmake g++ make clang
RUN git clone https://github.com/google/benchmark.git
RUN mkdir build
WORKDIR /app/build
RUN cmake ../benchmark -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
RUN make -j4
RUN make install

WORKDIR /app
VOLUME /app
