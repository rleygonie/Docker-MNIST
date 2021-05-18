Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/bionic64"
  config.vm.network "forwarded_port", guest:5000, host:5000, host_ip: "127.0.0.1"
  config.vagrant.plugins = "vagrant-docker-compose"
  config.vm.provision :docker
  config.vm.provision :docker_compose, yml: "/vagrant/docker-compose.yml", run:"always"
end
