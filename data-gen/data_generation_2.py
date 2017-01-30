import torchfile

if __name__ == '__main__':
    # From DeepMind.
    with open('labels.txt', 'w') as f:
        for i in range(1, 100001):
            o = torchfile.load('./mnist-cluttered/output/label_{}.t7'.format(i))
            label = o.argmax()
            f.write('{}\t{}\n'.format(i, label))
            print(i)
