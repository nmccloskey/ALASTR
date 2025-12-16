import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from backend.tools.logger import logger
from spacy import displacy
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from backend.nlp.NLPmodel import NLPmodel
from PyPDF2 import PdfMerger
import tempfile
from tempfile import NamedTemporaryFile
from sklearn.preprocessing import StandardScaler


def add_grouping_columns(df, combinations):
    """
    Adds grouping columns to DataFrame by combining values from multiple columns.

    Args:
        df (pd.DataFrame): The labeled DataFrame.
        combinations (list[tuple[str]]): List of column combinations.

    Returns:
        pd.DataFrame: Modified DataFrame with new grouping columns added.
    """
    for combo in combinations:
        if len(combo) > 1:
            combo = [c for c in combo]
            logger.info(f"combo: {combo}")
            col_name = "_".join(combo)
            df[col_name] = df[combo].astype(str).agg("_".join, axis=1)
    return df

def get_subplot_grid(n):
    """
    Determines optimal subplot grid (rows, cols) for n plots.

    Returns:
        tuple: (n_rows, n_cols)
    """
    n_cols = math.ceil(math.sqrt(n))
    n_rows = math.ceil(n / n_cols)
    return n_rows, n_cols

def make_density_plot(OM, df, feature, grouping_cols, plot_title, file_path, max_groups=10):
    """
    Creates a grid of density plots for each group_by column.
    Skips plots with too many unique group values.

    Args:
        OM: OutputManager with .save_image().
        df (pd.DataFrame): Data containing group columns and target feature.
        feature (str): Name of the column to plot.
        grouping_cols (list[str]): Grouping column names for subplot grid.
        plot_title (str): Title for the whole plot grid.
        file_path (str): Output path for saving figure.
        max_groups (int): Maximum number of groups allowed in a KDE plot.
    """
    try:
        n_plots = len(grouping_cols)
        n_rows, n_cols = get_subplot_grid(n_plots)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for i, group_col in enumerate(grouping_cols):
            ax = axes[i]
            n_groups = df[group_col].nunique()

            if n_groups > max_groups:
                logger.warning(f"Too many groups ({n_groups}) in '{group_col}' - using boxplot fallback.")
                try:
                    sns.boxplot(
                        data=df, x=group_col, y=feature, ax=ax
                    )
                    ax.set_title(f"Boxplot of {feature} by {group_col}")
                except Exception as box_err:
                    logger.warning(f"Boxplot fallback failed for '{group_col}': {box_err}")
                    ax.set_title(f"{group_col} (error)")
                    ax.axis("off")
                continue

            try:
                sns.kdeplot(
                    data=df, x=feature, hue=group_col,
                    fill=True, common_norm=False, alpha=0.5,
                    ax=ax
                )
                ax.set_title(f"Grouped by {group_col}")
                ax.set_ylabel("Density")
            except Exception as plot_err:
                logger.warning(f"Failed to plot feature '{feature}' grouped by '{group_col}': {plot_err}")
                ax.set_title(f"{group_col} (error)")
                ax.axis("off")

        for j in range(n_plots, len(axes)):
            axes[j].axis("off")

        plt.suptitle(plot_title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        OM.save_image(file_path, fig)
        plt.close(fig)

    except Exception as e:
        logger.error(f"Failed to plot density grid for '{feature}': {e}")

def make_pairplot(OM, df, plot_title, file_path, min_unique=2, max_nan_frac=0.3):
    """
    Generates and saves a seaborn pairplot with KDE in the lower triangle.
    Skips columns with too many NaNs or insufficient unique values.

    Args:
        OM: Output manager with a .save_image() method.
        df (pd.DataFrame): DataFrame of selected features.
        plot_title (str): Title for the plot.
        file_path (str): Output file path for the saved image.
        min_unique (int): Minimum number of unique values required to plot a column.
        max_nan_frac (float): Maximum fraction of NaNs allowed per column.
    """
    try:
        # Filter valid columns
        valid_cols = [
            col for col in df.columns
            if df[col].nunique(dropna=True) >= min_unique
            and df[col].isna().mean() <= max_nan_frac
        ]

        if len(valid_cols) < 2:
            logger.warning(f"Not enough valid features for pairplot: {len(valid_cols)} columns after filtering.")
            return

        logger.info(f"Creating pairplot with features: {valid_cols}")
        clean_df = df[valid_cols].dropna()

        sns.set_theme(style="whitegrid")
        pair_plot = sns.pairplot(clean_df)

        # KDE in lower triangle only if safe
        try:
            pair_plot.map_lower(sns.kdeplot, levels=4, color=".2")
        except Exception as kde_err:
            logger.warning(f"Lower-triangle KDE plot failed: {kde_err}")

        pair_plot.figure.suptitle(plot_title, y=1.02)
        fig = pair_plot.figure

        OM.save_image(file_path, fig)
        plt.close(fig)

    except Exception as e:
        logger.error(f"Failed to generate pairplot '{plot_title}': {e}")

def visualize_distinctive_features(OM, section, comparison_cols):
    """
    Generates KDE plots and pairplots for the most distinctive features in a section,
    based on maximum Cohen's d between groups.

    Args:
        OM: Project/output manager with .tables and visualization utilities.
        section (str): Section name for which to process tables.
    """
    logger.info(f"Visualizing section {section}.")

    for table in OM.tables.values():
        if table.section != section or "raw" not in table.tags:
            continue

        logger.info(f"Visualizing table {table.name}")

        labeled_df = OM.get_data_with_groupings(
            fact_table=table.grouping_table,
            dim_table=table.name,
            cluster_table=f"{table.name}_clusters" if OM.cluster else None,
            grouping_cols=comparison_cols
        )
        if labeled_df is None:
            logger.info(f"No data for table {table.name}")
            continue

        combinations = OM.eda.get_grouping_combos(comparison_cols, "gcomp")
        grouping_cols = ["_".join(combo) for combo in combinations] + [None]
        logger.info(f"Grouping cols: {grouping_cols}")
        labeled_df = add_grouping_columns(labeled_df, combinations)

        max_val_df = OM.eda.get_distinctive_features(table)
        dist_feats = set()

        for row in max_val_df.to_dict(orient="records"):
            feature = row["feature"]
            if feature in dist_feats:
                continue

            if labeled_df[feature].dropna().shape[0] < 3:
                logger.warning(f"Not enough valid values for feature '{feature}' — skipping.")
                continue

            filtered_df = labeled_df.dropna(subset=[feature])

            # kdeplot_title = f"{feature}_{int(row['group_name_1'])}v{int(row['group_name_2'])}"
            # kdeplot_file_path = os.path.join(
            #     table.get_file_path(),
            #     "distinctive_features", "density_plots", table.name,
            #     f"{kdeplot_title}_kdeplots.png"
            # )

            # make_density_plot(
            #     OM=OM, df=filtered_df, feature=feature,
            #     grouping_cols=grouping_cols,
            #     plot_title=kdeplot_title,
            #     file_path=kdeplot_file_path
            # )
            dist_feats.add(feature)

            if len(dist_feats) >= OM.max_feature_visuals:
                break

        if len(dist_feats) > 1:
            pairplot_title = f"Top {len(dist_feats)} Distinctive Features in {table.name}"
            pairplot_file_path = os.path.join(
                table.get_file_path(),
                "distinctive_features", "pairplots",
                f"{table.name}_dist_feat_pairplot.png"
            )

            valid_feats = [f for f in dist_feats if f in labeled_df.columns]
            select_df = labeled_df[valid_feats].dropna()

            make_pairplot(
                OM=OM, df=select_df,
                plot_title=pairplot_title,
                file_path=pairplot_file_path
            )

def make_spacy_dep_pdfs(doc, doc_id: str, path: str):
    """
    Generate and merge dependency tree PDFs for all sentences in a SpaCy Doc.

    Args:
        doc (spacy.tokens.Doc): Parsed document with sentence boundaries.
        doc_id (str): Unique ID to name the merged PDF file.
        path (str): Output directory for dependency tree PDFs.

    Notes:
        - Uses SpaCy displaCy visualizer.
        - Uses svglib + reportlab for SVG -> PDF conversion.
        - Merges all sentence PDFs into one per document.
    """
    NLP = NLPmodel()
    nlp = NLP.get_nlp()
    os.makedirs(path, exist_ok=True)

    logger.info(f"Generating dependency tree PDF for doc: {doc_id}")
    
    pdf_paths = []

    for i, sent in enumerate(doc.sents, 1):
        subdoc = nlp(sent.text)
        svg = displacy.render(subdoc, style="dep", page=False, jupyter=False)

        try:
            with NamedTemporaryFile(delete=False, suffix=".svg") as tmp_svg:
                tmp_svg_path = tmp_svg.name
                tmp_svg.write(svg.encode("utf-8"))

            drawing = svg2rlg(tmp_svg_path)
            tmp_pdf_path = NamedTemporaryFile(delete=False, suffix=".pdf").name
            renderPDF.drawToFile(drawing, tmp_pdf_path)
            pdf_paths.append(tmp_pdf_path)

        except Exception as e:
            logger.error(f"Failed to render dep tree for sentence {i}: {e}")
        finally:
            try:
                os.remove(tmp_svg_path)
            except Exception as cleanup_err:
                logger.warning(f"Could not delete temp SVG: {cleanup_err}")

    # Merge all per-sentence PDFs
    if pdf_paths:
        merged_path = os.path.join(path, f"doc_{doc_id}_dep_trees.pdf")
        merger = PdfMerger()
        for pdf in pdf_paths:
            merger.append(pdf)
        merger.write(merged_path)
        merger.close()

        logger.info(f"Merged dependency tree PDF saved: {merged_path}")

        # Cleanup temp PDFs
        for pdf in pdf_paths:
            try:
                os.remove(pdf)
            except Exception as e:
                logger.warning(f"Could not delete temp PDF: {e}")

def generate_corr_maps(OM, section: str, image_format: str = "pdf"):
    """
    Generate and save correlation heatmaps for all 'raw' tables within a given section.
    
    Args:
        OM: An object manager with `tables` and `save_image()` method.
        section (str): Section name to filter relevant tables.
        image_format (str): Output image format ('pdf' or 'png'). Defaults to 'pdf'.

    Notes:
        - Only numerical columns are used.
        - Annotation is auto-disabled for large matrices (>=100 cells).
        - Figure size scales dynamically based on number of features.
    """
    logger.info(f"Making corr maps for section {section}.")

    sheet_names, corr_mats, file_paths, granularities = [], [], [], []
    for table in OM.tables.values():
        if table.section != section or "raw" not in table.tags:
            continue

        logger.info(f"Generating heatmap for table {table.name}")
        df = table.get_data()

        if df is None or df.empty:
            logger.warning(f"Table {table.name} is empty.")
            continue

        df = df.drop(columns=table.get_pks(), errors="ignore")
        num_df = df.select_dtypes(include=[np.number])

        if num_df.empty:
            logger.warning(f"Table {table.name} has no numerical data for corr map.")
            continue

        corr_mat = num_df.corr()

        # Dynamic annotation toggle
        annot = corr_mat.size < 100  # Disable if 100+ cells

        # Dynamic figure size
        n_vars = corr_mat.shape[0]
        fig_width = min(20, max(6, n_vars * 0.6))
        fig_height = min(15, max(5, n_vars * 0.5))

        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(corr_mat, annot=annot, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title(f"Feature Correlations for {table.name}")
        plt.tight_layout()

        file_path = os.path.join(OM.output_dir, table.subdir, "correlations")
        os.makedirs(file_path, exist_ok=True)
        image_path = os.path.join(file_path, f"{table.name}_corr_heatmap.{image_format}")
        OM.save_image(image_path, plt)
        plt.close()

        sheet_names.append(table.name)
        corr_mats.append(corr_mat)
        file_paths.append(file_path)
        granularities.append(table.granularity)

    for sheet_name, corr_mat, file_path, gran in zip(sheet_names, corr_mats, file_paths, granularities):
        gran_spec = f"_{gran}" if gran else ""
        df_path = os.path.join(file_path, f"{section}{gran_spec}_corr_matrices.xlsx")
        
        if corr_mat is not None and not corr_mat.empty:
            mode = "a" if os.path.exists(df_path) else "w"
            with pd.ExcelWriter(df_path, mode=mode) as writer:
                corr_mat.to_excel(writer, sheet_name=sheet_name, index=False)
            logger.info(f"Exported corr map for table '{sheet_name}' to {file_path}")
        else:
            logger.warning(f"Dataframe for '{sheet_name}' is empty. Skipping export.") 

def generate_data_heatmaps(OM, section: str, chunk_size: int = 50):
    logger.info(f"Generating raw data heatmaps for section {section}")
    clustermap_index_dict = {}

    for table in OM.tables.values():
        if table.section != section or "raw" not in table.tags:
            continue

        logger.info(f"Visualizing raw data in table {table.name}")
        df = table.get_data()
        if df is None or df.empty:
            continue

        pk_cols = table.get_pks()
        df_pks = df[pk_cols] if pk_cols else pd.DataFrame(index=df.index)
        df_num = df.drop(columns=pk_cols, errors="ignore").select_dtypes(include=[np.number])
        if df_num.empty:
            logger.warning(f"No numeric data in table {table.name}")
            continue

        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_num), columns=df_num.columns, index=df.index)
        df_scaled.replace([np.inf, -np.inf], np.nan, inplace=True)

        df_clustermap = df_scaled.dropna(axis=0, how="any")

        # Output directory
        output_dir = os.path.join(OM.output_dir, table.subdir, "heatmaps")
        os.makedirs(output_dir, exist_ok=True)

        if df_clustermap is not None and not df_clustermap.empty:
            # --- Generate clustermap
            clustermap = sns.clustermap(df_clustermap, cmap="viridis", figsize=(10, 10), yticklabels=False)
            clustermap_path = tempfile.mktemp(suffix="_clustermap.pdf")
            clustermap.savefig(clustermap_path)
            plt.close()

            # --- Get reordered indices and match with PKs
            reordered_idx = df_scaled.index[clustermap.dendrogram_row.reordered_ind]
            if not df_pks.empty:
                reordered_pks = df_pks.loc[reordered_idx]
                clustermap_index_dict[table.name] = reordered_pks.reset_index(drop=True)
            temp_files = [clustermap_path]
        else:
            logger.warning(f"No clustermap possible for {table.name}.")
            temp_files = []

        # --- Generate chunked heatmaps
        for start in range(0, len(df_scaled), chunk_size):
            chunk = df_scaled.iloc[start:start+chunk_size]
            fig_width = max(10, chunk.shape[1] * 0.6)
            fig_height = max(5, chunk.shape[0] * 0.25)

            plt.figure(figsize=(fig_width, fig_height))
            sns.heatmap(chunk, cmap="gnuplot2", linewidths=0.3, vmin=-2, vmax=2)
            plt.title(f"{table.name} Rows {start + 1} - {start + len(chunk)}")
            plt.tight_layout()

            tmp_path = tempfile.mktemp(suffix=".pdf")
            plt.savefig(tmp_path)
            temp_files.append(tmp_path)
            plt.close()

        # --- Merge PDF
        merged_pdf_path = os.path.join(output_dir, f"{table.name}_data_heatmap.pdf")
        merger = PdfMerger()
        for path in temp_files:
            merger.append(path)
        merger.write(merged_pdf_path)
        merger.close()
        logger.info(f"Saved heatmap PDF: {merged_pdf_path}")

        for path in temp_files:
            try:
                os.remove(path)
            except Exception as e:
                logger.warning(f"Could not delete temp file: {e}")

    # --- Export all reordered indices to Excel
    if clustermap_index_dict:
        excel_path = os.path.join(OM.output_dir, f"{section}_clustermap_indices.xlsx")
        with pd.ExcelWriter(excel_path) as writer:
            for table_name, pk_df in clustermap_index_dict.items():
                pk_df.to_excel(writer, sheet_name=table_name, index=False)
        logger.info(f"Saved reordered indices Excel file: {excel_path}")
