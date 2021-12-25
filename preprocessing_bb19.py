import os
import sys
import json
import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('root')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--token', type=str, default='train', required=True, choices=['train', 'dev'])
    parser.add_argument('--input_dir', type=str, default='../dataset/BioNLP-OST-2019-BB-rel+ner_train', required=True)
    parser.add_argument('--output_dir', type=str, default=None, required=True)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, f"process_bb19_{args.token}.log"), 'w'))

    logger.info(sys.argv)
    logger.info(args)

    filename_list = os.listdir(args.input_dir)
    doc_list = list({_.split('.')[0] for _ in filename_list})
    logger.info(f'{len(doc_list)} docs for {args.token}.')

    output_path = os.path.join(args.output_dir, args.token + '.json')
    logger.info(f'output to {output_path}')
    with open(output_path, 'w') as output_file:
        for doc_id in doc_list:
            raw_text_path = os.path.join(args.input_dir, f'{doc_id}.txt')
            logger.info(f'Loading raw text from {raw_text_path}.')
            with open(raw_text_path, 'r') as raw_text_file:
                raw_text = raw_text_file.read()

            # load ground truth entities and relations
            ground_truth_path = os.path.join(args.input_dir, f'{doc_id}.a2')
            logger.info(f'Loading ground truth file from {ground_truth_path}.')
            ground_truth_entities = []
            ground_truth_relations = []
            with open(ground_truth_path, 'r') as ground_truth_file:
                for line in ground_truth_file:
                    if line[0] == 'T':
                        ent_id, ent_type_pos, ent_name = line.strip().split('\t')
                        try:
                            ent_type, pos_a, pos_b = ent_type_pos.split(' ')
                        except:
                            continue
                        ground_truth_entities.append({
                            'ent_id': ent_id,
                            'ent_name': ent_name,
                            'ent_type': ent_type,
                            'ent_pos': (pos_a, pos_b)
                        })
                    elif line[0] == 'R':
                        rel_id, rel_type_edge = line.strip().split('\t')
                        rel_type, edge_s, edge_t = rel_type_edge.split(' ')
                        ground_truth_relations.append({
                            'rel_id': rel_id,
                            'rel_type': rel_type,
                            'rel_edge': (edge_s, edge_t)
                        })

            # output
            structured_doc = {
                'doc_key': doc_id,
                'sentences': [],
                'ner': [[]],
                'relations': [[]]
            }

            # entity
            sent_wds = raw_text.split(' ')
            structured_doc['sentences'].append(sent_wds[:250])
            ent_id2wd_id = {}
            for entity in ground_truth_entities:
                pos_a, pos_b = entity['ent_pos']
                pos_a, pos_b = int(pos_a), int(pos_b)

                begin_wid = None
                end_wid = -1

                cur_wd_len_sum = 0
                for wid, wd in enumerate(sent_wds):
                    if cur_wd_len_sum >= pos_a and cur_wd_len_sum + len(wd) <= pos_b:
                        if begin_wid is None:
                            begin_wid = wid
                        end_wid = max(end_wid, wid)
                    cur_wd_len_sum += len(wd) + 1
                if begin_wid is not None and begin_wid <= end_wid:
                    structured_doc['ner'][0].append([begin_wid, end_wid, entity['ent_type']])
                    ent_id2wd_id[entity['ent_id']] = (begin_wid, end_wid)

            # relation
            for rel in ground_truth_relations:
                edge_s, edge_t = rel['rel_edge']
                ent_s = edge_s.split(':')[-1]
                ent_t = edge_t.split(':')[-1]
                if ent_s in ent_id2wd_id and ent_t in ent_id2wd_id:
                    bound_s = ent_id2wd_id[ent_s]
                    bound_t = ent_id2wd_id[ent_t]
                    structured_doc['relations'][0].append([bound_s[0], bound_s[1], bound_t[0], bound_t[1], rel['rel_type']])

            output_file.write(json.dumps(structured_doc) + '\n')
