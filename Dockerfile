FROM debian:10.8
RUN apt-get update
RUN apt-get -y install \
	git \
	build-essential \
	python3-numpy python3-scipy python3-matplotlib python3-pip \
	ninja-build
RUN pip3 install meson jupyterlab pytry
RUN git clone https://github.com/nengo/nengo \
	&& cd nengo \
	&& pip3 install -e . \
	&& cd ..
RUN git clone https://github.com/astoeckel/libbioneuronqp \
	&& cd libbioneuronqp \
	&& git checkout ebbc0e2a81e2c6f1fef89ec8765022f1bb1f5eff \
	&& git submodule init \
	&& git submodule update \
	&& mkdir build \
	&& cd build \
	&& meson .. \
	&& ninja install \
	&& cd .. \
	&& pip3 install -e . \
	&& cd ..
RUN git clone https://github.com/astoeckel/nengo-bio \
	&& cd nengo-bio \
	&& git checkout 8ff2274f8f5e09877ed0a554ca268f9fa396e3a5 \
	&& pip3 install -e . \
	&& cd ..
EXPOSE 4567/TCP
ENV LD_LIBRARY_PATH=/libbioneuronqp/build/
CMD cd /cogsci2020-cerebellum && jupyter-lab --allow-root --port 4567 --ip 0.0.0.0 --no-browser
