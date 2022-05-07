<template>
    <div id="data-content">
        <div id="confusion-matrix-container">
            <confusion-matrix ref="matrix" id="confusion-matrix" :showColor="true" :confusionMatrix="confusionMatrix">
            </confusion-matrix>
        </div>
    </div>
</template>

<script>
import ConfusionMatrix from './ConfusionMatrix.vue';
import axios from 'axios';

export default {
    components: {ConfusionMatrix},
    name: 'DataView',
    data() {
        return {
            confusionMatrix: undefined,
        };
    },
    methods: {
    },
    mounted: function() {
        const store = this.$store;
        const that = this;
        axios.post(store.getters.URL_GET_CONFUSION_MATRIX)
            .then(function(response) {
                that.confusionMatrix = response.data;
            });
    },
};
</script>

<style scoped>
#data-content {
    width: 100%;
    height: 100%;
    display: flex;
}

#confusion-matrix-container {
  width: 100%;
  height: 100%;
  display: flex;
  margin: 10px 10px 10px 10px;
}
</style>
