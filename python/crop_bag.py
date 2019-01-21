import rosbag
import argparse

            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--num_msgs", required= True)
    parser.add_argument("-i","--input", required= True)
    parser.add_argument("-o","--output", required= True)
    
    args = parser.parse_args()
    num_msgs = args.num_msgs
    with rosbag.Bag(args.output, 'w') as outbag:
        for topic, msg, t in rosbag.Bag(args.input).read_messages():
            while num_msgs:
                outbag.write(topic, msg, t)
                num_msgs -= 1


if __name__ == '__main__':
    main()
              