"""
Main script – Run the coin detection pipeline.
CORRECTIF : Gère automatiquement la différence entre "grp" (dans le CSV) et "gp" (dossiers réels).

Usage:
    python main.py                      # process all images + evaluate
    python main.py data/gp1/image.jpg   # process a single image
    python main.py --visualize          # process all + open result windows
"""

import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend
import matplotlib.pyplot as plt

from coin_detector import CoinDetector

# ============================================================
# CONFIG
# ============================================================

DATA_DIR      = Path(__file__).parent / 'data'
GT_FILE       = Path(__file__).parent / 'ground_truth.csv'
RESULTS_DIR   = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================
# GROUND TRUTH LOADER
# ============================================================

def load_ground_truth(gt_path):
    """
    Load ground truth from CSV.
    Returns dict: filename -> {'num_coins': int, 'value_eur': float|None, 'group': str}
    """
    gt = {}
    if not gt_path.exists():
        return gt
        
    with open(gt_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row['filename'].strip()
            grp = row.get('group', '').strip() 
            
            nc_str = row.get('num_coins', '').strip()
            try:
                nc = int(nc_str) if nc_str and nc_str.lower() != 'nan' else None
            except ValueError:
                nc = None
            
            val_str = row.get('value_eur', '').strip().replace(',', '.')
            try:
                ve = float(val_str) if val_str and val_str.lower() != 'nan' else None
            except ValueError:
                ve = None
            
            # On utilise le nom original du CSV pour la clé
            key = f"{grp}/{fn}" if grp else fn
            
            gt[key] = {
                'filename': fn,
                'num_coins': nc,
                'value_eur': ve,
                'group': grp,
            }
    return gt


def find_image_path(data_dir, filename, group=None):
    """
    Find the actual file on disk.
    Auto-corrects 'grp' to 'gp' if needed.
    """
    extensions = ['.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG']
    
    # Liste des dossiers potentiels à tester
    # 1. Le groupe tel quel (ex: gp1)
    # 2. Le groupe corrigé (grp1 -> gp1)
    # 3. Pas de groupe (racine data/)
    candidate_groups = []
    if group:
        candidate_groups.append(group)
        if 'grp' in group:
            candidate_groups.append(group.replace('grp', 'gp'))
        if 'gp' in group:
             # Au cas où le CSV dirait gp mais le dossier serait grp (peu probable mais bon)
            candidate_groups.append(group.replace('gp', 'grp'))
    
    # Recherche avec groupe
    for grp_candidate in candidate_groups:
        base_path = data_dir / grp_candidate / filename
        if base_path.exists():
            return base_path, grp_candidate # On retourne aussi le nom du dossier trouvé

        # Test des extensions
        stem = base_path.stem
        parent = base_path.parent
        for ext in extensions:
            alt = parent / (stem + ext)
            if alt.exists():
                return alt, grp_candidate

    # Recherche sans groupe (à la racine de data/)
    base_root = data_dir / filename
    if base_root.exists():
        return base_root, None
    
    # Extensions à la racine
    for ext in extensions:
        alt = data_dir / (base_root.stem + ext)
        if alt.exists():
            return alt, None
            
    return None, None

# ============================================================
# EVALUATION METRICS
# ============================================================

def evaluate(predictions, ground_truth):
    """Compute evaluation metrics."""
    count_errors = []
    value_errors = []
    exact_count = 0
    close_value = 0 
    total_eval = 0
    total_val_eval = 0

    per_group = {}

    for key, pred in predictions.items():
        if key not in ground_truth:
            continue
        gt = ground_truth[key]
        # On normalise le nom du groupe pour l'affichage des stats (tout en 'gp')
        grp = gt['group'].replace('grp', 'gp')
        
        if grp not in per_group:
            per_group[grp] = {'count_err': [], 'val_err': [], 'exact': 0, 'n': 0, 'n_val': 0, 'close_val': 0}

        if gt['num_coins'] is not None:
            ce = abs(pred['count'] - gt['num_coins'])
            count_errors.append(ce)
            per_group[grp]['count_err'].append(ce)
            per_group[grp]['n'] += 1
            total_eval += 1
            if ce == 0:
                exact_count += 1
                per_group[grp]['exact'] += 1

        if gt['value_eur'] is not None:
            ve = abs(pred['total_value'] - gt['value_eur'])
            value_errors.append(ve)
            per_group[grp]['val_err'].append(ve)
            per_group[grp]['n_val'] += 1
            total_val_eval += 1
            if ve <= 0.50:
                close_value += 1
                per_group[grp]['close_val'] += 1

    metrics = {
        'count_mae': np.mean(count_errors) if count_errors else None,
        'count_exact_rate': exact_count / total_eval if total_eval else None,
        'value_mae': np.mean(value_errors) if value_errors else None,
        'value_close_rate': close_value / total_val_eval if total_val_eval else None,
        'n_images': total_eval,
        'per_group': {},
    }

    for grp, d in per_group.items():
        metrics['per_group'][grp] = {
            'count_mae': np.mean(d['count_err']) if d['count_err'] else None,
            'count_exact_rate': d['exact'] / d['n'] if d['n'] else None,
            'value_mae': np.mean(d['val_err']) if d['val_err'] else None,
            'value_close_rate': d['close_val'] / d['n_val'] if d['n_val'] else None,
            'n': d['n'],
        }

    return metrics

# ============================================================
# REPORTING
# ============================================================

def print_metrics(metrics):
    """Pretty-print evaluation results."""
    print('\n' + '=' * 65)
    print('  EVALUATION RESULTS')
    print('=' * 65)
    print(f"  Images evaluated        : {metrics['n_images']}")
    if metrics['count_mae'] is not None:
        print(f"  Count MAE               : {metrics['count_mae']:.2f}")
        print(f"  Count exact-match rate  : {metrics['count_exact_rate']:.1%}")
    if metrics['value_mae'] is not None:
        print(f"  Value MAE (EUR)         : {metrics['value_mae']:.2f}")
        print(f"  Value within ±0.50 EUR  : {metrics['value_close_rate']:.1%}")

    sorted_groups = sorted(metrics['per_group'].keys())
    for grp in sorted_groups:
        gm = metrics['per_group'][grp]
        print(f"\n  --- Group: {grp} ({gm['n']} images) ---")
        if gm['count_mae'] is not None:
            print(f"      Count MAE          : {gm['count_mae']:.2f}")
            print(f"      Exact count rate   : {gm['count_exact_rate']:.1%}")
        if gm['value_mae'] is not None:
            print(f"      Value MAE (EUR)    : {gm['value_mae']:.2f}")
            print(f"      Close value rate   : {gm['value_close_rate']:.1%}")
    print('=' * 65 + '\n')


def save_detailed_csv(predictions, ground_truth, path):
    """Save per-image results to CSV."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            'group', 'filename',
            'gt_count', 'pred_count', 'count_error',
            'gt_value', 'pred_value', 'value_error',
            'denominations_detected',
        ])
        sorted_keys = sorted(predictions.keys()) 
        
        for key in sorted_keys:
            pred = predictions[key]
            gt = ground_truth.get(key, {})
            
            filename = gt.get('filename', key)
            grp  = gt.get('group', '?')
            # Correction pour l'affichage CSV
            if grp and 'grp' in grp: grp = grp.replace('grp', 'gp')

            gt_c = gt.get('num_coins', '')
            gt_v = gt.get('value_eur', '')
            
            pc = pred['count']
            pv = pred['total_value']
            ce = abs(pc - gt_c) if isinstance(gt_c, (int, float)) else ''
            ve = round(abs(pv - gt_v), 2) if isinstance(gt_v, (int, float)) else ''
            denoms = ', '.join(c.get('denomination', '?') for c in pred.get('coins', []))
            
            w.writerow([grp, filename, gt_c, pc, ce, gt_v, pv, ve, denoms])
    print(f"  Detailed results saved to {path}")


def create_summary_figure(predictions, ground_truth, path):
    """Create a summary bar chart."""
    keys = sorted([k for k in predictions if k in ground_truth
                      and ground_truth[k]['num_coins'] is not None])
    if not keys:
        return

    gt_counts = [ground_truth[k]['num_coins'] for k in keys]
    pr_counts = [predictions[k]['count'] for k in keys]
    
    labels = []
    for k in keys:
        g = ground_truth[k].get('group', '')
        if g and 'grp' in g: g = g.replace('grp', 'gp') # Correction label
        f = ground_truth[k].get('filename', '')
        labels.append(f"{g}/{f}" if g else f)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # --- Count comparison ---
    x = np.arange(len(keys))
    w = 0.35
    axes[0].bar(x - w/2, gt_counts, w, label='Ground Truth', color='steelblue')
    axes[0].bar(x + w/2, pr_counts, w, label='Predicted', color='coral')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=90, ha='center', fontsize=6)
    axes[0].set_ylabel('Number of coins')
    axes[0].set_title('Coin Count: Ground Truth vs Predicted')
    axes[0].legend()

    # --- Value comparison ---
    keys_v = [k for k in keys if ground_truth[k].get('value_eur') is not None]
    if keys_v:
        gt_vals = [ground_truth[k]['value_eur'] for k in keys_v]
        pr_vals = [predictions[k]['total_value'] for k in keys_v]
        labels_v = []
        for k in keys_v:
            g = ground_truth[k].get('group', '')
            if g and 'grp' in g: g = g.replace('grp', 'gp')
            f = ground_truth[k].get('filename', '')
            labels_v.append(f"{g}/{f}" if g else f)
            
        x2 = np.arange(len(keys_v))
        axes[1].bar(x2 - w/2, gt_vals, w, label='Ground Truth', color='steelblue')
        axes[1].bar(x2 + w/2, pr_vals, w, label='Predicted', color='coral')
        axes[1].set_xticks(x2)
        axes[1].set_xticklabels(labels_v, rotation=90, ha='center', fontsize=6)
        axes[1].set_ylabel('Value (EUR)')
        axes[1].set_title('Monetary Value: Ground Truth vs Predicted')
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"  Summary figure saved to {path}")

# ============================================================
# MAIN
# ============================================================

def process_single_image(image_path):
    """Process a single image and display the result."""
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"  ERROR: image not found: {image_path}")
        return

    print('\n' + '=' * 65)
    print('  EURO COIN DETECTION (Single Mode)')
    print('=' * 65)

    detector = CoinDetector(target_width=800, use_knn=True)
    print(f'  Processing: {image_path.name}\n')

    t0 = time.time()
    result = detector.process_image(image_path)
    dt = time.time() - t0

    # Print results
    print(f"  Coins detected : {result['count']}")
    print(f"  Total value    : {result['total_value']:.2f} EUR")
    print(f"  Processing time: {dt:.2f}s\n")

    if result.get('coins'):
        print(f"  {'#':<4s} {'Denomination':<14s} {'Colour':<12s} {'Value':>6s}")
        print('  ' + '-' * 40)
        for i, c in enumerate(result['coins'], 1):
            denom = c.get('denomination', '?')
            color = c.get('color_group', '?')
            val   = c.get('value', 0)
            print(f"  {i:<4d} {denom:<14s} {color:<12s} {val:>5.2f} EUR")
        print()

    # Save visualisation
    if 'image' in result and result['image'] is not None:
        # Essayer de déduire le groupe du dossier parent
        possible_group = image_path.parent.name
        # Normalisation grp -> gp
        if 'grp' in possible_group: possible_group = possible_group.replace('grp', 'gp')
        
        out_dir = RESULTS_DIR / possible_group
        out_dir.mkdir(parents=True, exist_ok=True)
        
        vis = detector.visualize(result['image'], result, title=image_path.name)
        out_path = out_dir / f"result_{image_path.name}"
        cv2.imwrite(str(out_path), vis)
        print(f"  Result image saved to {out_path}")

    print('=' * 65 + '\n')


def main():
    show_viz = '--visualize' in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    
    if args:
        process_single_image(args[0])
        return

    print('\n' + '=' * 65)
    print('  EURO COIN DETECTION (Batch Mode)')
    print('=' * 65)

    if not GT_FILE.exists():
        print(f"ERROR: ground truth file not found: {GT_FILE}")
        return
    gt = load_ground_truth(GT_FILE)
    print(f"  Loaded {len(gt)} ground-truth entries.")

    detector = CoinDetector(target_width=800, use_knn=True)
    print('  Detector initialised.\n')

    predictions = {}
    processing_times = []

    for key, gt_data in sorted(gt.items()):
        fn = gt_data['filename']
        csv_group = gt_data['group']
        
        # Recherche de l'image (auto-correction grp -> gp incluse)
        img_path, found_group = find_image_path(DATA_DIR, fn, group=csv_group)
        
        if img_path is None:
            print(f"  [SKIP] {csv_group}/{fn:20s}  image not found")
            continue

        # Utiliser le nom du dossier réel (gp1) s'il a été trouvé, sinon corriger manuellement
        final_group_name = found_group if found_group else csv_group.replace('grp', 'gp')

        t0 = time.time()
        result = detector.process_image(img_path)
        dt = time.time() - t0
        processing_times.append(dt)

        predictions[key] = result

        gt_c = str(gt_data['num_coins']) if gt_data['num_coins'] is not None else '?'
        gt_v = f"{gt_data['value_eur']:.2f}" if gt_data['value_eur'] is not None else '?'
        pc = result['count']
        pv = result['total_value']
        
        status = 'OK' if str(pc) == gt_c else 'MISS'
        display_name = f"{final_group_name}/{fn}"
        print(f"  [{status:4s}] {display_name:25s}  count: {pc:2d}/{gt_c:<3s}  "
              f"val: {pv:5.2f}/{gt_v:<6s}")

        # --- SAUVEGARDE IMAGE DANS DOSSIER GROUPE ---
        if 'image' in result and result['image'] is not None:
            vis = detector.visualize(result['image'], result, title=f"{final_group_name} - {fn}")
            
            # On utilise final_group_name (ex: gp3) pour créer le dossier
            group_out_dir = RESULTS_DIR / final_group_name
            group_out_dir.mkdir(parents=True, exist_ok=True)
            
            out_path = group_out_dir / f"result_{img_path.name}"
            
            cv2.imwrite(str(out_path), vis)

            if show_viz:
                cv2.imshow(f"Result: {final_group_name}/{fn}", vis)
                cv2.waitKey(500)

    if show_viz:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if processing_times:
        print(f"\n  Avg processing time: {np.mean(processing_times):.2f}s per image")

    metrics = evaluate(predictions, gt)
    print_metrics(metrics)
    save_detailed_csv(predictions, gt, RESULTS_DIR / 'detailed_results.csv')
    create_summary_figure(predictions, gt, RESULTS_DIR / 'summary_comparison.png')

    print('\n  Done! Check the "results/" folder for outputs.\n')


if __name__ == '__main__':
    main()