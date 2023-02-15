import torch
import numpy as np
from lib.test_dataLoader import ParseData
import lib.utils as utils
import os
import torch_geometric as ptg

def get_test_predictions(model, dataloader, device, args):
    dataloader = ParseData(args.dataset,suffix=args.suffix,mode=args.mode, args=args)
    test_encoder, test_graph, test_batch, joint_times, node_times = dataloader.load_data(
            sample_percent=args.sample_percent_test,
            batch_size=1,data_type="test")

    t = torch.tensor(joint_times, device=device)
    all_dts = t.unsqueeze(1) - t.unsqueeze(2) # (B, N_T, N_T)
    # Index [:, i, j] is (t_j - t_i), time from t_i to t_j
    off_diags = [torch.diagonal(all_dts, offset=offset, dim1=1, dim2=2).t()
            for offset in range(args.max_pred+1)]
    # List of length max_preds, each entry is tensor: (diag_length, B)
    padded_off_diags = torch.nn.utils.rnn.pad_sequence(off_diags,
            batch_first=False, padding_value=-1.0) # (N_T, max_pred+1, B)

    pred_delta_times = padded_off_diags[:,1:].transpose(1,2).transpose(0,1)
    # (B, N_T, max_pred)
    # Index [:, i, j] is (t_(i+j) - t_i), time from t_i to t_(i+j)

    all_preds_list = []
    for sample_i, (batch_dict_encoder, batch_dict_graph, sample_t,\
        sample_node_ts, sample_pred_times) in enumerate(zip(
            test_encoder, test_graph, t, node_times, pred_delta_times)):
        print(f"Evaluating sample {sample_i}")
        st_graph_ts = 1.5 + batch_dict_encoder.pos
        sample_node_ts = [np.array(ts) for ts in sample_node_ts]

        sample_predictions = []
        for cur_t, pred_ts in zip(sample_t, sample_pred_times):
            # pred_ts is (max_pred,)
            # True if node in the st-graph is before time cur_t
            st_t_mask = st_graph_ts-1e-4  <= cur_t # Small delta for numerics
            new_y = torch.LongTensor([np.sum(ts-1e-4 <= cur_t.item())
                for ts in sample_node_ts]).to(device)

            # Create subgraph of node obs <= cur_t
            sub_edge_index, sub_edge_attr = ptg.utils.subgraph(st_t_mask,
                    batch_dict_encoder.edge_index,
                    edge_attr=batch_dict_encoder.edge_attr,
                    relabel_nodes=True, num_nodes=args.n_balls)
            _, sub_edge_same = ptg.utils.subgraph(st_t_mask,
                    batch_dict_encoder.edge_index,
                    edge_attr=batch_dict_encoder.edge_same,
                    relabel_nodes=True, num_nodes=args.n_balls)

            # Adjust to be relative to cur_t
            new_pos = (st_graph_ts - cur_t)[st_t_mask] - 0.5 # NOTE: Important

            new_data = ptg.data.Data(
                    batch = batch_dict_encoder.batch[st_t_mask],
                    edge_attr = sub_edge_attr,
                    edge_index = sub_edge_index,
                    edge_same = sub_edge_same,
                    pos = new_pos,
                    x = batch_dict_encoder.x[st_t_mask],
                    y = new_y
                ).to(device)


            # Adjust for padded sequences, values do not matter, but must work with ODE
            padding_mask = (pred_ts == -1)
            n_padding = torch.sum(padding_mask)
            pred_ts[padding_mask] = 1.0 + torch.linspace(0.,1.,n_padding, device=device)

            # This is all decoder batch is
            batch_dict_decoder = {"time_steps": pred_ts}
            pred_y, info, temporal_weights = model.get_reconstruction(new_data,
                    batch_dict_decoder,
                    batch_dict_graph.to(device),
                    n_traj_samples = 1) # Fix to one predictive sample
            # pred_y is [n_traj=1, N, n_time_steps, d_y=2]

            reshaped_pred = pred_y[0,:,:,0] # (N, max_pred)
            sample_predictions.append(reshaped_pred)

        sample_predictions_tensor = torch.stack(sample_predictions,
                dim=0) # (N_T, N, max_pred)
        all_preds_list.append(sample_predictions_tensor)

    all_predictions = torch.stack(all_preds_list, dim=0) # (N_data, N_T, N, max_pred)

    print("Saving predictions")
    os.makedirs("predictions", exist_ok=True)
    save_name = os.path.join("predictions", f"{args.data}_{args.load}.pt")
    torch.save(all_predictions, save_name)
    print("Done!")

