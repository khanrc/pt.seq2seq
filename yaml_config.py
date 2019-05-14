import yaml


class YAMLConfig:
    def __init__(self, path, argv=None):
        if not path.endswith('.yaml'):
            path += '.yaml'

        self.path = path
        # dictionary
        self._cfg = yaml.load(open(path), Loader=yaml.Loader)
        if argv:
            self.parse_update(argv)

    def str(self):
        # string w/o comments
        return yaml.dump(self._cfg, sort_keys=False).strip()

    def markdown(self):
        # markdown
        return self.str().replace(' ', '&nbsp;').replace('\n', '  \n')

    def parse_update(self, argv):
        """ parse argv & update self._cfg
        argv structure: [option1, value1, option2, value2, ...]
        """
        N = len(argv)
        assert N % 2 == 0

        for i in range(0, N, 2):
            opts = argv[i]
            value = argv[i+1]
            assert opts.startswith('--')
            opts = opts[2:].split('.')
            dic = self._cfg
            for opt in opts[:-1]:
                dic = dic[opt]
            last_opt = opts[-1]
            assert last_opt in dic, "{} is not the element of {}".format(last_opt, self.path)
            type_cls = type(dic[last_opt])
            dic[last_opt] = type_cls(value)

    def __repr__(self):
        return self.str()

    def __getitem__(self, key):
        return self._cfg[key]


if __name__ == "__main__":
    import sys
    config = YAMLConfig(sys.argv[1])
    print(config.str())
    print("---")
    argv = sys.argv[3:]
    print("argv:", argv)
    config.parse_update(argv)
    print(config.str())
