<template>
    <div id="data-content">
        <div id="confusion-matrix-container">
            <confusion-matrix v-if="matrixType==='confusion'" ref="matrix" id="confusion-matrix" :showColor="true" :confusionMatrix="confusionMatrix">
            </confusion-matrix>
            <numerical-matrix v-else-if="matrixType==='numerical'" ref="numerical"
                :numericalMatrix="numericalMatrix" :numericalMatrixType="numericalMatrixType"></numerical-matrix>
        </div>
    </div>
</template>

<script>
import ConfusionMatrix from './ConfusionMatrix.vue';
import NumericalMatrix from './NumericalMatrix.vue';
import axios from 'axios';

export default {
    components: {ConfusionMatrix, NumericalMatrix},
    name: 'DataView',
    data() {
        return {
            matrixType: 'numerical', // confusion, numerical
            numericalMatrixType: 'size',
            confusionMatrix: undefined,
            numericalMatrix: undefined,
        };
    },
    methods: {
        getMatrix: function() {
            const store = this.$store;
            const that = this;
            if (this.matrixType==='confusion') {
                axios.post(store.getters.URL_GET_CONFUSION_MATRIX)
                    .then(function(response) {
                        that.confusionMatrix = response.data;
                    });
            } else if (this.matrixType==='numerical') {
                if (this.numericalMatrixType==='size') {
                    axios.post(store.getters.URL_GET_SIZE_MATRIX)
                        .then(function(response) {
                            const numericalMatrix = response.data;
                            for (let i=0; i<numericalMatrix.partitions.length; i++) {
                                numericalMatrix.partitions[i] = Math.round(numericalMatrix.partitions[i]*100)/100;
                            }
                            numericalMatrix.partitions.unshift('N');
                            that.numericalMatrix = numericalMatrix;
                        });
                }
            }
        },
    },
    mounted: function() {
        this.getMatrix();
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
