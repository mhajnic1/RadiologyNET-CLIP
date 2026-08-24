"""Demo interface for the fine-tuned retrieval model.

Local only on purpose. The gallery serves real patient scans, so this must never be
launched with share=True or bound to anything other than localhost.

    python app.py
"""

import gradio as gr

from src.app.gallery import Gallery

gallery = Gallery()

INTRO = f"""# RadiologyNET-CLIP retrieval demo

Fine-tuned BiomedCLIP, checkpoint `{gallery.checkpoint}`.
Gallery holds {len(gallery.meta):,} images across {gallery.meta.ExamID.nunique():,} exams.

The task is the first tab, diagnosis to image. The second tab runs the reverse direction,
which the same training produces for free because CLIP's loss is symmetric. It is kept as
a check that the shared embedding space is genuinely aligned rather than as a feature.

Queries must be in English, and the model was trained on full structured reports rather
than short phrases, so a few keywords will retrieve noticeably worse than a real
diagnosis will. That gap is expected on a dataset this size and is part of what the
demo shows.
"""


def search_images(query, top_k, test_only, dedupe):
    query = (query or '').strip()
    if not query:
        return [], 'Type a diagnosis, or pick one from the dropdown.'

    results, n_correct = gallery.text_to_image(query, int(top_k), test_only, dedupe)
    items = []
    for rank, (row, sim, is_correct) in enumerate(results, 1):
        info = gallery.row_info(row)
        views = f", {info['n_views']} views" if info['n_views'] > 1 else ''
        tick = ' [match]' if is_correct else ''
        items.append((gallery.views(row)[0],
                      f"{rank}. {info['modality']} {sim:.3f}{views}{tick}"))

    note = f"{len(results)} results, scope: {'test split' if test_only else 'full dataset'}."
    if n_correct:
        hits = sum(1 for _, _, c in results if c)
        note += f" This query is a real diagnosis with {n_correct} matching image(s); {hits} returned."
    else:
        note += ' Free-text query, so there is no ground truth to check against.'
    return items, note


def show_random(test_only):
    row = gallery.sample_rows(1, test_only)[0]
    return row, *render_image(row)


def render_image(row):
    info = gallery.row_info(row)
    caption = (f"id {info['id']} | {info['modality']} | exam {info['exam']} "
               f"| {info['split']} split | {info['n_views']} view(s)")
    return gallery.views(row), caption, info['diagnosis']


def search_text(row, top_k, test_only):
    if row is None:
        return '', 'Pick an image first.'
    results, truth = gallery.image_to_text(int(row), int(top_k), test_only)
    lines = []
    for rank, (text, sim, is_correct) in enumerate(results, 1):
        mark = ' **[correct]**' if is_correct else ''
        lines.append(f"**{rank}.** ({sim:.3f}){mark}\n\n{text}\n\n---")
    return '\n'.join(lines), f'True diagnosis for this image:\n\n{truth}'


with gr.Blocks(title='RadiologyNET-CLIP') as demo:
    gr.Markdown(INTRO)

    with gr.Tab('Diagnosis to image'):
        with gr.Row():
            with gr.Column(scale=2):
                query = gr.Textbox(label='Diagnosis', lines=4,
                                   placeholder='Paste or type a diagnosis in English')
                picker = gr.Dropdown(choices=gallery.test_diagnoses(),
                                     label='Or pick a real test diagnosis',
                                     value=None, filterable=True)
                picker.change(lambda d: d or '', inputs=picker, outputs=query)
            with gr.Column(scale=1):
                k1 = gr.Slider(1, 20, value=10, step=1, label='Results')
                test1 = gr.Checkbox(False, label='Test split only (model never saw these)')
                dedupe = gr.Checkbox(True, label='One image per exam')
                go1 = gr.Button('Search', variant='primary')

        note1 = gr.Markdown()
        out1 = gr.Gallery(label='Retrieved images', columns=5, height='auto')
        go1.click(search_images, [query, k1, test1, dedupe], [out1, note1])

    with gr.Tab('Image to diagnosis'):
        with gr.Row():
            with gr.Column(scale=1):
                test2 = gr.Checkbox(False, label='Test split only')
                pick = gr.Button('Pick a random image', variant='primary')
                row_state = gr.Number(label='Row index', precision=0)
                k2 = gr.Slider(1, 20, value=10, step=1, label='Results')
                go2 = gr.Button('Find diagnoses')
            with gr.Column(scale=2):
                img_out = gr.Gallery(label='Query image (all views)', columns=4, height='auto')
                img_cap = gr.Markdown()

        truth_out = gr.Markdown()
        out2 = gr.Markdown()

        pick.click(show_random, [test2], [row_state, img_out, img_cap, truth_out])
        go2.click(search_text, [row_state, k2, test2], [out2, truth_out])


if __name__ == '__main__':
    # localhost only, no share link, patient data must not leave this machine
    demo.launch(server_name='127.0.0.1', share=False, inbrowser=True)
