def enable_tidal(
    model,
    attn_type="tidal",
    top_k=256,
    sparse_layer_start=2,
    correction_layer=13,
    attention_sink=0,
    lim_ratio=1,
):
    if attn_type == "tidal":
        print(f"TidalDecode Enabled: attention_sink: {attention_sink}")
        print(f"token budget: {top_k}")
        print(f"sparse layer starts from: Layer {sparse_layer_start}")
        print(f"reselection layer: {correction_layer}")
        model_type = model.config.model_type

        if "llama" in model_type:
            from src.tidal_build.modify_llama import (
                enable_llama_tidal_attention,
            )

            enable_llama_tidal_attention(
                model,
                top_k,
                attn_type,
                sparse_layer_start,
                correction_layer,
            )
        elif "qwen" in model_type:
            # currently support qwen family
            from src.tidal_build.modify_qwen3 import (
                enable_qwen_tidal_attention,
            )

            enable_qwen_tidal_attention(
                model,
                top_k,
                attn_type,
                sparse_layer_start,
                correction_layer,
                # attention_sink,
                # lim_ratio,
            )
    elif attn_type == "lim":
        print(f"LessIsMore Enabled: attention_sink: {attention_sink}")
        print(f"token budget: {top_k}")
        print(f"sparse layer starts from: Layer {sparse_layer_start}")
        print(f"reselection layer: {correction_layer}")
        model_type = model.config.model_type

        if "llama" in model_type:
            from src.tidal_build.modify_llama_lim import (
                enable_llama_tidal_attention,
            )

            enable_llama_tidal_attention(
                model,
                top_k,
                attn_type,
                sparse_layer_start,
                correction_layer,
                attention_sink,
                lim_ratio,
            )
        elif "qwen" in model_type:
            # currently support qwen family
            from src.tidal_build.modify_qwen3_lim import (
                enable_qwen_tidal_attention,
            )

            enable_qwen_tidal_attention(
                model,
                top_k,
                attn_type,
                sparse_layer_start,
                correction_layer,
                attention_sink,
                lim_ratio,
            )
    return
