import sys
from pathlib import Path
import pandas as pd
import numpy as np
import re

sys.path.append(str((Path(__file__).parent / '../..')))

from torchai.utils import parse_log_record, SequenceCollector


def read_log(path: Path) -> pd.DataFrame:
    with path.open() as logfile:
        parsed_records = (parse_log_record(record) for record in logfile.readlines())
        valid_records = (record for record in parsed_records if len(record) > 2)
        df = pd.DataFrame.from_records(valid_records)

    for column in df.columns:
        df[column] = pd.to_numeric(df[column])
    return df


def read_all_logs(path: Path):
    return {
        file.stem: read_log(file)
        for file in path.iterdir()
    }


def extract_layers_sizes(log_name: str):
    pattern = r'D-([^_]*)'
    sizes_string = re.findall(pattern, log_name)[0]
    return [int(i) for i in sizes_string.split('-')]


def compute_stats(log_df: pd.DataFrame):
    solved_score = 200
    longest_duration = 0
    longest_episode = 0
    solved_episode = -1

    fail_collector = SequenceCollector(lambda x: x < 50)
    solved_collector = SequenceCollector(lambda x: x > solved_score)

    for _, record in log_df.iterrows():
        if longest_duration < record.Duration:
            longest_duration = record.Duration
            longest_episode = record.Episode
        if record['Mean duration'] > solved_score:
            solved_episode = record.Episode
        fail_collector.push(record.Duration)
        solved_collector.push(record['Mean duration'])
    fail_collector.close()
    solved_collector.close()

    if len(fail_collector.sequences) > 0:
        longest_fails_sequence_length = max(len(seq) for seq in fail_collector.sequences)
        mean_fails_sequence_length = np.mean([len(seq) for seq in fail_collector.sequences])
    else:
        longest_fails_sequence_length = -1
        mean_fails_sequence_length = 0

    if len(solved_collector.sequences) > 0:
        longest_solved_sequence_length = max(len(seq) for seq in solved_collector.sequences)
    else:
        longest_solved_sequence_length = -1

    return {
        'duration_mean': float(np.mean(log_df[['Duration']])),
        'duration_median': float(np.median(log_df[['Duration']])),
        'duration_std': float(np.std(log_df[['Duration']])),
        'longest_episode': int(longest_episode),
        'solved_episode': int(solved_episode),
        'largest_score': float(np.max(log_df[['Mean duration']])),
        'longest_duration': float(longest_duration),
        'longest_fails_seq': longest_fails_sequence_length,
        'mean_fails_seq': mean_fails_sequence_length,
        'longest_solved_seq': longest_solved_sequence_length,
    }


def main():
    path = Path(sys.argv[1])
    logs = read_all_logs(path)
    for name, df in logs.items():
        print("{}: {}".format(name, compute_stats(df)))


if __name__ == '__main__':
    main()
