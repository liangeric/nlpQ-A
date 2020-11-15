SHARE=$(pwd)
IMAGE=$1
NAME=$(sudo docker run -d -v ${SHARE}:/host/Users -it ${IMAGE} /bin/bash)
echo '****************'
sudo docker exec -it $NAME ./ask /host/Users/data/set1/a1.txt 3
echo '****************'
sudo docker exec -it $NAME ./answer /host/Users/data/set1/a1.txt /host/Users/test_questions.txt
echo '****************'
sudo docker stop $NAME >/dev/null
