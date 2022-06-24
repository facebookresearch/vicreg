from argparse import ArgumentParser
import yaml
from sklearn.model_selection import ParameterGrid


def main():
    with open(args.path_to_grid, 'r') as obj:
        params_list = yaml.safe_load(obj)
    # iterate on experiment to create experiment grid
    experiments_dict = {}
    with open(args.path_to_save, 'w') as obj:
        obj.write('##### START EXPERIMENTS #####')
    for experiment, values in params_list['experiments'].items():
        param_grid = {
            key: val if isinstance(val, list) else [val] for key, val in values.items() if key != 'default_argument'
        }
        grid = ParameterGrid(param_grid)
        command_list = []
        for val in grid:
            command_list.append(' '.join([f'--{param} {value}' for param, value in val.items()] +
                                         [f'--{param}' for param in values['default_argument']
                                          ]
                                         )
                                )
        experiments_dict[experiment] = command_list
        with open(args.path_to_save, 'a') as obj:
            obj.write(f'\n## EXPERIMENT {experiment}\n')
            obj.write('\n'.join(command_list))
    print('\n'.join([str(x) for x in experiments_dict.values()]))


if __name__ == '__main__':
    parser = ArgumentParser(description='create sbatch txt and run ograrray', allow_abbrev=False)
    parser.add_argument('--path_to_save', default='sbatches/grid_sbatches.txt')
    parser.add_argument('--path_to_grid', type=str,
                        help='List of tuples containing parameters on which you want to grid. Tuple structure:'
                             '(parameter, start_value, ending_value, number of steps)',
                        default='sbatches/params_to_grid.yaml')
    args = parser.parse_args()
    main()
